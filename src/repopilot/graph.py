
from typing import Any, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from repopilot.candidate_generator import CandidateGenerator
from repopilot.content_retriever import ContentRetriever
from repopilot.file_selector import FileSelection, FileSelector
from repopilot.repository_analyzer import RepositoryAnalyzer
from repopilot.repository_context import (
    RepositoryContext,
    RepositoryContextBuilder,
)
from repopilot.repository_discovery import RepositoryDiscoverer
from repopilot.guardrails.input_validation import validate_request
from repopilot.guardrails.output_validation import validate_output

MAX_RECOVERY_ATTEMPTS = 1

class EvidenceAssessment(BaseModel):
    sufficient: bool = Field(
        description=(
            "True only if the supplied repository context contains "
            "enough direct evidence to answer the user's question."
        )
    )
    missing_information: str = Field(
        description=(
            "Brief explanation of any evidence still missing, "
            "or why the evidence is sufficient."
        )
    )


class RepoPilotState(TypedDict, total=False):
    """Serializable workflow data passed between graph nodes."""

    owner: str
    repository: str
    query: str
    model_name: str

    snapshot: Any
    analysis: Any
    candidates: list
    selection_candidates: list
    selection: FileSelection
    selected_paths: list[str]
    attempted_paths: list[str]
    retrieved_files: list
    context: RepositoryContext

    recovery_attempts: int
    evidence_sufficient: bool
    evidence_reason: str

    answer: str
    status: str
    error: str
    evidence_assessment_failed: bool
    evidence_assessment_error: str
    input_errors: list[str]


def build_graph(
    tools: list,
    model_name: str = "llama3.2:3b",
):
    """Build and compile the RepoPilot LangGraph workflow."""

    analyzer = RepositoryAnalyzer()
    generator = CandidateGenerator(max_candidates=20)

    selector = FileSelector(
        model_name=model_name,
        temperature=0.0,
    )

    retriever = ContentRetriever(
        max_files=10,
        max_file_chars=30_000,
        max_total_chars=100_000,
    )

    context_builder = RepositoryContextBuilder(
        max_total_chars=50_000,
    )

    async def discover_repository(state: RepoPilotState) -> dict:
        discoverer = RepositoryDiscoverer(
            tools=tools,
            owner=state["owner"],
            repository=state["repository"],
        )
        snapshot = await discoverer.discover()
        return {"snapshot": snapshot}

    def analyze_repository(state: RepoPilotState) -> dict:
        return {
            "analysis": analyzer.analyze(state["snapshot"]),
        }

    def generate_candidates(state: RepoPilotState) -> dict:
        candidates = generator.generate(
            snapshot=state["snapshot"],
            analysis=state["analysis"],
            query=state["query"],
        )

        return {
            "candidates": candidates,
            "selection_candidates": candidates,
            "attempted_paths": [],
            "retrieved_files": [],
            "recovery_attempts": 0,
        }

    async def select_files(state: RepoPilotState) -> dict:
        selection = await selector.select(
            query=state["query"],
            candidates=state["selection_candidates"],
        )
        return {"selection": selection}

    def validate_selection(state: RepoPilotState) -> dict:
        selection = selector.validate_selection(
            selection=state["selection"],
            candidates=state["selection_candidates"],
        )

        paths = list(dict.fromkeys(
            item.path for item in selection.selected_files
        ))

        return {
            "selection": selection,
            "selected_paths": paths,
        }

    async def retrieve_files(state: RepoPilotState) -> dict:
        paths = state["selected_paths"]

        new_files = await retriever.retrieve(
            tools=tools,
            owner=state["owner"],
            repo=state["repository"],
            paths=paths,
        )

        # Preserve the first retrieval when recovery adds more files.
        existing_files = state.get("retrieved_files", [])
        merged_files = list(existing_files)
        known_paths = {file.path for file in merged_files}

        for file in new_files:
            if file.path not in known_paths:
                merged_files.append(file)
                known_paths.add(file.path)

        attempted = list(state.get("attempted_paths", []))
        attempted.extend(paths)
        attempted = list(dict.fromkeys(attempted))

        return {
            "retrieved_files": merged_files,
            "attempted_paths": attempted,
        }

    def build_context(state: RepoPilotState) -> dict:
        return {
            "context": context_builder.build(
                state.get("retrieved_files", [])
            ),
        }

    
    async def check_evidence(state: RepoPilotState) -> dict:
        context = state["context"]

        # No usable content: recovery may find better source files.
        if not context.text.strip():
            reason = (
                "No usable source content was retrieved."
                if context.failed_paths
                else "No usable source content was assembled."
            )
            return {
                "evidence_sufficient": False,
                "evidence_reason": reason,
                "evidence_assessment_failed": False,
                "evidence_assessment_error": "",
            }

        try:
            llm = ChatOllama(
                model=state.get("model_name", "llama3.2:3b"),
                temperature=0.0,
            )
            assessor = llm.with_structured_output(EvidenceAssessment)

            assessment = await assessor.ainvoke([
                SystemMessage(
                    content=(
                        "You are RepoPilot's repository evidence assessor. "
                        "Decide whether the supplied source code contains enough "
                        "direct evidence to answer the user's actual question.\n\n"

                        "Use these rules:\n"
                        "1. Focus on the user's central question, not every related "
                        "detail that could be investigated.\n"
                        "2. Mark evidence as sufficient if the code supports a useful, "
                        "factually grounded answer to the central question.\n"
                        "3. For broad questions, a high-level explanation of the "
                        "supported flow is sufficient. Do not require every layer, "
                        "such as middleware, templates, models, authentication, "
                        "or error handling, unless the question specifically asks "
                        "about those layers.\n"
                        "4. If the question has multiple parts, assess whether the "
                        "main parts can be answered from the evidence. Missing "
                        "nonessential details must not make the entire assessment "
                        "insufficient.\n"
                        "5. Mark evidence as insufficient only when a material "
                        "part of the central question cannot be answered reliably "
                        "from the supplied code, or when the supplied content is "
                        "irrelevant or too incomplete.\n"
                        "6. Evaluate actual code and relationships between files, "
                        "not filenames alone.\n"
                        "7. Never invent implementation details. A qualified or "
                        "partial answer is acceptable when the available evidence "
                        "supports it.\n\n"

                        "Return sufficient=True when the central question can be "
                        "answered with the available evidence, even if some "
                        "secondary details remain unknown. In missing_information, "
                        "briefly explain why the evidence is sufficient or identify "
                        "the specific material information that is missing."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Question:\n{state['query']}\n\n"
                        f"Included files:\n"
                        f"{', '.join(context.included_paths)}\n\n"
                        f"Repository evidence:\n{context.text}"
                    )
                ),
            ])

            # Validate the structured response before trusting it.
            if not isinstance(assessment, EvidenceAssessment):
                assessment = EvidenceAssessment.model_validate(assessment)

            return {
                "evidence_sufficient": assessment.sufficient,
                "evidence_reason": assessment.missing_information,
                "evidence_assessment_failed": False,
                "evidence_assessment_error": "",
            }

        except Exception as exc:
            # Keep the exception visible in state, but don't expose a
            # traceback or internal details in the user-facing answer.
            return {
                "evidence_sufficient": False,
                "evidence_reason": "Evidence assessment could not be completed.",
                "evidence_assessment_failed": True,
                "evidence_assessment_error": (
                    f"{type(exc).__name__}: {exc}"
                ),
            }

    
    def route_evidence(state: RepoPilotState) -> str:
        if state.get("evidence_assessment_failed", False):
            if state.get("recovery_attempts", 0) >= MAX_RECOVERY_ATTEMPTS:
                return "assessment_failed"

            attempted = set(state.get("attempted_paths", []))
            remaining = [
                candidate
                for candidate in state.get("candidates", [])
                if candidate.path not in attempted
            ]

            return "recover" if remaining else "assessment_failed"

        if state.get("evidence_sufficient", False):
            return "answer"

        if state.get("recovery_attempts", 0) >= MAX_RECOVERY_ATTEMPTS:
            return "insufficient"

        attempted = set(state.get("attempted_paths", []))
        remaining = [
            candidate
            for candidate in state.get("candidates", [])
            if candidate.path not in attempted
        ]

        return "recover" if remaining else "insufficient"

    async def recover_files(state: RepoPilotState) -> dict:
        attempted = set(state.get("attempted_paths", []))

        remaining = [
            candidate
            for candidate in state.get("candidates", [])
            if candidate.path not in attempted
        ]

        # Defensive guard: the graph's routing should already ensure
        # that candidates remain.
        if not remaining:
            return {
                "selection_candidates": [],
                "selected_paths": [],
                "recovery_attempts": (
                    state.get("recovery_attempts", 0) + 1
                ),
                "evidence_assessment_failed": False,
                "evidence_assessment_error": "",
            }

        return {
            "selection_candidates": remaining,
            "recovery_attempts": (
                state.get("recovery_attempts", 0) + 1
            ),
            "evidence_assessment_failed": False,
            "evidence_assessment_error": "",
        }

    async def answer_query(state: RepoPilotState) -> dict:
        context = state["context"]

        # Safety check: never generate a normal answer when the
        # evidence assessment says it is insufficient.
        if not context.text.strip():
            return {
                "answer": (
                    "I couldn't find usable repository content to answer "
                    "this question reliably."
                ),
                "status": "insufficient_evidence",
            }

        llm = ChatOllama(
            model=state.get("model_name", "llama3.2:3b"),
            temperature=0.0,
        )

        try:
            response = await llm.ainvoke([
                SystemMessage(
                    content=(
                        "You are RepoPilot, a repository analysis agent. "
                        "Answer using only the supplied repository context. "
                        "Cite exact file paths that support your claims. "
                        "Explain what the code does, not merely what filenames "
                        "suggest. Do not invent files, functions, or "
                        "implementation details. Clearly distinguish facts "
                        "from any uncertainty."
                    )
                ),
                HumanMessage(
                    content=(
                        f"Repository: {state['owner']}/"
                        f"{state['repository']}\n\n"
                        f"User question:\n{state['query']}\n\n"
                        f"Repository context:\n{context.text}"
                    )
                ),
            ])
        except Exception as exc:
            return {
                "answer": (
                    "RepoPilot found repository evidence, but couldn't "
                    "generate the final answer because the language "
                    "model encountered an error. Please try again."
                ),
                "status": "answer_generation_failed",
                "error": f"{type(exc).__name__}: {exc}",
            }

        metadata = getattr(response, "response_metadata", None) or {}
        usage = getattr(response, "usage_metadata", None) or {}

        print("\n[LLM Diagnostics: answer_query]")
        print(f"Model: {state.get('model_name', 'llama3.2:3b')}")
        print(f"Prompt tokens: {metadata.get('prompt_eval_count')}")
        print(f"Output tokens: {metadata.get('eval_count')}")
        print(
            "Prompt evaluation: "
            f"{metadata.get('prompt_eval_duration', 0) / 1e9:.2f}s"
        )
        print(
            "Generation: "
            f"{metadata.get('eval_duration', 0) / 1e9:.2f}s"
        )
        print(f"Usage metadata: {usage}")

        try:
            answer = response.content
            print("\n====================Printing response=====================\n")
            print(response)
            print("\n")
        except Exception:
            return {
                "answer": (
                    "RepoPilot received an invalid response from the "
                    "language model and cannot return it as a completed answer."
                ),
                "status": "invalid_model_output",
                "error": (
                    "The model response did not contain readable text content."
                ),
            }
        
        print("\n[LLM Response Content Diagnostics]")
        print(f"Response type: {type(response).__name__}")
        print(f"Content type: {type(answer).__name__}")

        if isinstance(answer, str):
            print(f"Content length: {len(answer)}")
            print(f"Content preview: {answer[:200]!r}")
        else:
            print(f"Content representation: {repr(answer)[:300]}")

        additional_kwargs = (
            getattr(response, "additional_kwargs", None) or {}
        )
        print(
            "Additional response fields: "
            f"{list(additional_kwargs.keys())}"
        )

        for key, value in additional_kwargs.items():
            if isinstance(value, str):
                print(f"{key}: type=str, length={len(value)}")
            elif isinstance(value, list):
                print(f"{key}: type=list, length={len(value)}")
            else:
                print(f"{key}: type={type(value).__name__}")
        
        print(f"Finish reason: {metadata.get('done_reason')}")
        print(f"Content is empty: {not isinstance(answer, str) or not answer.strip()}")

        output_errors = validate_output(answer)

        if output_errors:
            return {
                "answer": (
                    "RepoPilot generated an invalid response and "
                    "cannot return it as a completed answer. "
                    + " ".join(output_errors)
                ),
                "status": "invalid_model_output",
                "error": "; ".join(output_errors),
            }

        return {
            "answer": answer,
            "status": "completed",
        }

    def finish_insufficient_evidence(state: RepoPilotState) -> dict:
        context = state.get("context")
        included = context.included_paths if context else []
        failed = context.failed_paths if context else []

        reason = state.get(
            "evidence_reason",
            "The available repository context does not answer the question.",
        )

        if not included and failed:
            status = "retrieval_failed"
            explanation = (
                "Repository files could not be retrieved successfully."
            )
        else:
            status = "insufficient_evidence"
            explanation = (
                "The available repository evidence is insufficient."
            )

        details = (
            f"\nFiles examined: {', '.join(included)}"
            if included else ""
        )

        return {
            "answer": (
                f"{explanation} {reason}{details} "
                "I won't infer unsupported implementation details."
            ),
            "status": status,
        }
    

    def finish_assessment_failed(state: RepoPilotState) -> dict:
        context = state.get("context")
        included = context.included_paths if context else []
        error = state.get(
            "evidence_assessment_error",
            "Unknown assessment error",
        )

        files_examined = (
            f" Files examined: {', '.join(included)}."
            if included
            else ""
        )

        return {
            "answer": (
                "I retrieved repository context, but could not reliably "
                "assess whether it answers your question. I won't present "
                "an unverified answer as established fact."
                f"{files_examined}"
            ),
            "status": "evidence_assessment_failed",
            "error": error,
        }

    def finish_no_candidates(state: RepoPilotState) -> dict:
        return {
            "answer": (
                "No candidate files were identified for the question. "
                "Try a more specific query or inspect the repository "
                "snapshot and candidate-generation signals."
            ),
            "status": "no_candidates",
        }

    def finish_no_selection(state: RepoPilotState) -> dict:
        return {
            "answer": (
                "The file selector did not produce any valid candidate "
                "paths. No source files were retrieved."
            ),
            "status": "no_valid_selection",
        }

    def validate_input(state: RepoPilotState) -> dict:
        errors = validate_request(
            owner=state.get("owner"),
            repository=state.get("repository"),
            query=state.get("query"),
            model_name=state.get("model_name", model_name),
        )
        return {"input_errors": errors}

    def route_input(state: RepoPilotState) -> str:
        return "invalid" if state.get("input_errors") else "valid"

    def finish_invalid_input(state: RepoPilotState) -> dict:
        errors = state.get("input_errors", [])
        return {
            "answer": (
                "The request was rejected because its input is invalid. "
                + " ".join(errors)
            ),
            "status": "invalid_input",
        }
    
    def route_candidates(state: RepoPilotState) -> str:
        return "select" if state.get("candidates") else "no_candidates"

    def route_selection(state: RepoPilotState) -> str:
        if state.get("selected_paths"):
            return "retrieve"

        # An empty recovery selection should reach the normal
        # sufficiency/fallback routing, not the initial-selection error.
        if state.get("recovery_attempts", 0) > 0:
            return "check_evidence"

        return "no_selection"

    builder = StateGraph(RepoPilotState)

    builder.add_node("discover_repository", discover_repository)
    builder.add_node("analyze_repository", analyze_repository)
    builder.add_node("generate_candidates", generate_candidates)
    builder.add_node("select_files", select_files)
    builder.add_node("validate_selection", validate_selection)
    builder.add_node("retrieve_files", retrieve_files)
    builder.add_node("build_context", build_context)
    builder.add_node("check_evidence", check_evidence)
    builder.add_node("recover_files", recover_files)
    builder.add_node("answer_query", answer_query)
    builder.add_node(
        "finish_insufficient_evidence",
        finish_insufficient_evidence,
    )
    builder.add_node("finish_no_candidates", finish_no_candidates)
    builder.add_node("finish_no_selection", finish_no_selection)
    builder.add_node(
        "finish_assessment_failed",
        finish_assessment_failed,
    )
    builder.add_node("validate_input", validate_input)
    builder.add_node("finish_invalid_input", finish_invalid_input)

    builder.add_edge(START, "validate_input")

    builder.add_conditional_edges(
        "validate_input",
        route_input,
        {
            "valid": "discover_repository",
            "invalid": "finish_invalid_input",
        },
    )

    builder.add_edge("finish_invalid_input", END)
    builder.add_edge("discover_repository", "analyze_repository")
    builder.add_edge("analyze_repository", "generate_candidates")

    builder.add_conditional_edges(
        "generate_candidates",
        route_candidates,
        {
            "select": "select_files",
            "no_candidates": "finish_no_candidates",
        },
    )

    builder.add_edge("select_files", "validate_selection")

    builder.add_conditional_edges(
        "validate_selection",
        route_selection,
        {
            "retrieve": "retrieve_files",
            "check_evidence": "check_evidence",
            "no_selection": "finish_no_selection",
        },
    )

    builder.add_edge("retrieve_files", "build_context")
    builder.add_edge("build_context", "check_evidence")

    builder.add_conditional_edges(
        "check_evidence",
        route_evidence,
        {
            "answer": "answer_query",
            "recover": "recover_files",
            "insufficient": "finish_insufficient_evidence",
            "assessment_failed": "finish_assessment_failed",
        },
    )

    builder.add_edge("recover_files", "select_files")

    builder.add_edge("answer_query", END)
    builder.add_edge("finish_insufficient_evidence", END)
    builder.add_edge("finish_no_candidates", END)
    builder.add_edge("finish_no_selection", END)
    builder.add_edge("finish_assessment_failed", END)

    return builder.compile()