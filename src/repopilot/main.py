
import argparse
import asyncio
import time
from typing import Any

from repopilot.github_mcp import github_mcp_session
from repopilot.graph import build_graph


PROGRESS_MESSAGES = {
    "discover_repository": "Discovering repository files...",
    "analyze_repository": (
        "Analyzing repository structure and technology signals..."
    ),
    "generate_candidates": "Generating relevant file candidates...",
    "select_files": "Selecting relevant source files...",
    "validate_selection": "Validating selected file paths...",
    "retrieve_files": "Retrieving source files...",
    "build_context": "Building repository context...",
    "check_evidence": (
        "Checking whether the retrieved evidence answers the question..."
    ),
    "recover_files": (
        "Evidence is incomplete or could not be assessed. "
        "Attempting bounded recovery..."
    ),
    "answer_query": (
        "Evidence accepted. Generating the final answer..."
    ),
    "finish_no_candidates": "No candidate files were found.",
    "finish_no_selection": "No valid files were selected.",
    "finish_insufficient_evidence": (
        "Unable to establish sufficient evidence."
    ),
    "finish_assessment_failed": "Evidence assessment failed.",
}


def format_progress_event(
    update: dict[str, Any],
    include_progress: bool = True,
) -> list[str]:
    """Convert LangGraph node updates into concise CLI messages."""
    messages = []

    for node_name, state_update in update.items():
        message = PROGRESS_MESSAGES.get(node_name)

        if message and include_progress:
            messages.append(message)

        if node_name == "retrieve_files":
            files = state_update.get("retrieved_files", [])

            if files:
                successful = sum(
                    1
                    for file in files
                    if file.status == "success"
                )
                failed = sum(
                    1
                    for file in files
                    if file.status == "failed"
                )

                messages.append(
                    f"Retrieval results: {successful} successful, "
                    f"{failed} failed."
                )

        elif node_name == "build_context":
            context = state_update.get("context")

            if context is not None:
                messages.append(
                    f"Context contains {len(context.included_paths)} "
                    f"files and {context.total_chars} characters."
                )

        elif node_name == "check_evidence":
            if state_update.get("evidence_assessment_failed"):
                messages.append(
                    "Evidence assessment encountered an error."
                )
            elif state_update.get("evidence_sufficient") is True:
                messages.append("Evidence is sufficient.")
            elif state_update.get("evidence_sufficient") is False:
                messages.append(
                    "Evidence is insufficient; "
                    "checking recovery options."
                )

    return messages


async def run(
    owner: str,
    repository: str,
    query: str,
    model_name: str,
) -> None:
    async with github_mcp_session() as tools:
        graph = build_graph(
            tools,
            model_name=model_name,
        )

        initial_state = {
            "owner": owner,
            "repository": repository,
            "query": query,
            "model_name": model_name,
        }

        # Accumulate node updates to preserve the final graph state.
        result: dict[str, Any] = {}

        # Track the start time of each running LangGraph task.
        task_start_times: dict[str, float] = {}

        # Measure total execution time for this request.
        request_start = time.perf_counter()

        print("\n=== RepoPilot Progress ===", flush=True)

        try:
            async for mode, event in graph.astream(
                initial_state,
                stream_mode=["updates", "tasks"],
            ):
                
                if mode == "tasks":
                    node_name = event.get("name", "unknown")
                    task_id = event.get("id", node_name)

                    # A task-start event contains its input.
                    if "input" in event:
                        task_start_times[task_id] = (
                            time.perf_counter()
                        )

                        message = PROGRESS_MESSAGES.get(
                            node_name,
                            f"Running {node_name}...",
                        )

                        print(
                            f"[RepoPilot] {message}",
                            flush=True,
                        )

                    # A task-result event contains a result or error.
                    elif "result" in event or "error" in event:
                        started_at = task_start_times.pop(
                            task_id,
                            None,
                        )

                        if started_at is not None:
                            elapsed = (
                                time.perf_counter() - started_at
                            )

                            error = event.get("error")
                            outcome = (
                                "failed" if error is not None
                                else "completed"
                            )

                            print(
                                f"[RepoPilot] {node_name} {outcome} "
                                f"in {elapsed:.2f} s",
                                flush=True,
                            )

                        if event.get("error") is not None:
                            print(
                                f"[RepoPilot] {node_name} failed. "
                                "Check diagnostic logs if needed.",
                                flush=True,
                            )


                elif mode == "updates":
                    # Preserve the existing state accumulation.
                    for node_update in event.values():
                        result.update(node_update)

                    # Preserve retrieval, context, and evidence
                    # summaries without duplicating progress messages.
                    for message in format_progress_event(
                        event,
                        include_progress=False,
                    ):
                        print(
                            f"[RepoPilot] {message}",
                            flush=True,
                        )

        except Exception:
            print(
                "[RepoPilot] Workflow execution failed. "
                "Check diagnostic logs if needed.",
                flush=True,
            )
            raise

        finally:
            total_elapsed = time.perf_counter() - request_start

            print(
                f"\n[RepoPilot] Total execution time: "
                f"{total_elapsed:.2f} s",
                flush=True,
            )

        print("\n=== RepoPilot Result ===")
        print(f"Status: {result.get('status', 'unknown')}")
        print(f"Question: {query}")

        context = result.get("context")

        if context is not None:
            print(f"Files used: {context.included_paths}")
            print(f"Context characters: {context.total_chars}")

        if result.get("error"):
            # This may contain internal diagnostic information.
            # Keep it out of normal CLI output for now.
            print("Diagnostic: An internal workflow error occurred.")

        print("\n=== Final Answer ===")
        print(result.get("answer", "No answer was produced."))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RepoPilot: LangGraph-powered repository analysis"
    )

    parser.add_argument("--owner", required=True)
    parser.add_argument("--repo", required=True)
    parser.add_argument("--query", required=True)
    parser.add_argument("--model", default="qwen3.5:2b")

    args = parser.parse_args()

    asyncio.run(
        run(
            owner=args.owner,
            repository=args.repo,
            query=args.query,
            model_name=args.model,
        )
    )


if __name__ == "__main__":
    main()
