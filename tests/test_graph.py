
from types import SimpleNamespace

import pytest

import repopilot.graph as graph_module
from repopilot.content_retriever import RetrievedFile
from repopilot.file_selector import FileSelection, SelectedFile
from repopilot.graph import EvidenceAssessment, build_graph


class FakeDiscoverer:
    def __init__(self, **kwargs):
        pass

    async def discover(self):
        return {"files": ["src/app.py", "src/routes.py"]}


class FakeAnalyzer:
    def analyze(self, snapshot):
        return {"framework": "generic"}


class FakeGenerator:
    candidates = [
        SimpleNamespace(path="src/app.py", score=1.0, reasons=["source"]),
        SimpleNamespace(path="src/routes.py", score=0.8, reasons=["routes"]),
    ]

    def __init__(self, **kwargs):
        pass

    def generate(self, **kwargs):
        return list(type(self).candidates)


class FakeSelector:
    def __init__(self, **kwargs):
        self.calls = 0

    async def select(self, query, candidates):
        self.calls += 1

        # Select the first candidate offered on each attempt.
        selected = candidates[0].path if candidates else None

        return FileSelection(
            selected_files=(
                [SelectedFile(path=selected, reason="Relevant source")]
                if selected
                else []
            )
        )

    def validate_selection(self, selection, candidates):
        valid_paths = {candidate.path for candidate in candidates}
        return FileSelection(
            selected_files=[
                item
                for item in selection.selected_files
                if item.path in valid_paths
            ]
        )


class FakeRetriever:
    results = {}

    def __init__(self, **kwargs):
        pass

    async def retrieve(self, tools, owner, repo, paths):
        return [
            self.results.get(
                path,
                RetrievedFile(
                    path=path,
                    status="failed",
                    error="Mocked retrieval failure",
                ),
            )
            for path in paths
        ]


class FakeLLM:
    assessments = []
    assessment_index = 0

    def __init__(self, **kwargs):
        pass

    def with_structured_output(self, schema):
        return self

    async def ainvoke(self, messages):
        # Evidence assessor calls return structured assessments.
        if FakeLLM.assessment_index < len(FakeLLM.assessments):
            result = FakeLLM.assessments[FakeLLM.assessment_index]
            FakeLLM.assessment_index += 1
            return result

        # The normal answer-generation call.
        return SimpleNamespace(
            content="The request flow is implemented in src/app.py."
        )


@pytest.fixture(autouse=True)
def fake_dependencies(monkeypatch):
    monkeypatch.setattr(
        graph_module, "RepositoryDiscoverer", FakeDiscoverer
    )
    monkeypatch.setattr(
        graph_module, "RepositoryAnalyzer", FakeAnalyzer
    )
    monkeypatch.setattr(
        graph_module, "CandidateGenerator", FakeGenerator
    )
    monkeypatch.setattr(
        graph_module, "FileSelector", FakeSelector
    )
    monkeypatch.setattr(
        graph_module, "ContentRetriever", FakeRetriever
    )
    monkeypatch.setattr(graph_module, "ChatOllama", FakeLLM)

    FakeLLM.assessments = []
    FakeLLM.assessment_index = 0

    FakeRetriever.results = {
        "src/app.py": RetrievedFile(
            path="src/app.py",
            content="def handle_request(): return route_request()",
            status="success",
        ),
        "src/routes.py": RetrievedFile(
            path="src/routes.py",
            content="def route_request(): return 'OK'",
            status="success",
        ),
    }


def make_state():
    return {
        "owner": "example",
        "repository": "sample",
        "query": "How does the request flow work?",
        "model_name": "fake-model",
    }


def assessment(sufficient, reason):
    return EvidenceAssessment(
        sufficient=sufficient,
        missing_information=reason,
    )


@pytest.mark.asyncio
async def test_sufficient_evidence_routes_to_answer():
    FakeLLM.assessments = [
        assessment(True, "The source directly explains the flow.")
    ]

    result = await build_graph(tools=[]).ainvoke(make_state())

    assert result["status"] == "completed"
    assert "src/app.py" in result["answer"]
    assert result["evidence_sufficient"] is True
    assert result["recovery_attempts"] == 0


@pytest.mark.asyncio
async def test_insufficient_evidence_triggers_one_recovery():
    FakeLLM.assessments = [
        assessment(False, "The routing implementation is missing."),
        assessment(True, "Both request handling and routing are present."),
    ]

    result = await build_graph(tools=[]).ainvoke(make_state())

    assert result["status"] == "completed"
    assert result["recovery_attempts"] == 1
    assert set(result["attempted_paths"]) == {
        "src/app.py",
        "src/routes.py",
    }
    assert set(result["context"].included_paths) == {
        "src/app.py",
        "src/routes.py",
    }


@pytest.mark.asyncio
async def test_recovery_preserves_original_retrieved_files():
    FakeLLM.assessments = [
        assessment(False, "More routing evidence is required."),
        assessment(True, "Combined evidence answers the question."),
    ]

    result = await build_graph(tools=[]).ainvoke(make_state())

    retrieved_paths = {
        item.path for item in result["retrieved_files"]
    }

    assert retrieved_paths == {"src/app.py", "src/routes.py"}
    assert "src/app.py" in result["context"].text
    assert "src/routes.py" in result["context"].text


@pytest.mark.asyncio
async def test_exhausted_recovery_returns_insufficient_evidence():
    FakeLLM.assessments = [
        assessment(False, "The initial evidence is incomplete."),
        assessment(False, "The additional evidence is still incomplete."),
    ]

    result = await build_graph(tools=[]).ainvoke(make_state())

    assert result["status"] == "insufficient_evidence"
    assert result["recovery_attempts"] == 1
    assert result["evidence_sufficient"] is False
    assert "insufficient" in result["answer"].lower()


@pytest.mark.asyncio
async def test_retrieval_failures_are_not_reported_as_missing_features():
    FakeRetriever.results = {
        path: RetrievedFile(
            path=path,
            status="failed",
            error="Simulated MCP timeout",
        )
        for path in ("src/app.py", "src/routes.py")
    }

    result = await build_graph(tools=[]).ainvoke(make_state())

    assert result["status"] == "retrieval_failed"
    assert "could not be retrieved" in result["answer"].lower()
    assert result["evidence_sufficient"] is False


@pytest.mark.asyncio
async def test_no_candidates_keeps_existing_status():
    FakeGenerator.candidates = []

    try:
        result = await build_graph(tools=[]).ainvoke(make_state())
        assert result["status"] == "no_candidates"
    finally:
        FakeGenerator.candidates = [
            SimpleNamespace(
                path="src/app.py", score=1.0, reasons=["source"]
            ),
            SimpleNamespace(
                path="src/routes.py", score=0.8, reasons=["routes"]
            ),
        ]


@pytest.mark.asyncio
async def test_no_valid_initial_selection_keeps_existing_status(monkeypatch):
    class EmptySelector(FakeSelector):
        async def select(self, query, candidates):
            return FileSelection(selected_files=[])

    monkeypatch.setattr(graph_module, "FileSelector", EmptySelector)

    result = await build_graph(tools=[]).ainvoke(make_state())

    assert result["status"] == "no_valid_selection"



@pytest.mark.asyncio
async def test_assessment_failure_recovers_and_retries():
    original_ainvoke = FakeLLM.ainvoke
    calls = 0

    async def fail_once_then_succeed(self, messages):
        nonlocal calls
        calls += 1

        if calls == 1:
            raise TimeoutError("Simulated Ollama timeout")

        if calls == 2:
            return assessment(True, "The recovered source answers the question.")

        return SimpleNamespace(
            content="The request flow is implemented in src/app.py."
        )

    FakeLLM.ainvoke = fail_once_then_succeed

    try:
        result = await build_graph(tools=[]).ainvoke(make_state())
    finally:
        FakeLLM.ainvoke = original_ainvoke

    assert result["status"] == "completed"
    assert result["recovery_attempts"] == 1
    assert result["evidence_assessment_failed"] is False
    assert result["evidence_assessment_error"] == ""
    assert set(result["context"].included_paths) == {
        "src/app.py",
        "src/routes.py",
    }


@pytest.mark.asyncio
async def test_assessment_failure_after_recovery_returns_distinct_status():
    original_ainvoke = FakeLLM.ainvoke
    calls = 0

    async def always_fail_assessment(self, messages):
        nonlocal calls
        calls += 1

        # Fail both evidence-assessment attempts.
        if calls <= 2:
            raise TimeoutError("Simulated Ollama timeout")

        return SimpleNamespace(content="Unexpected answer")

    FakeLLM.ainvoke = always_fail_assessment

    try:
        result = await build_graph(tools=[]).ainvoke(make_state())
    finally:
        FakeLLM.ainvoke = original_ainvoke

    assert result["status"] == "evidence_assessment_failed"
    assert result["recovery_attempts"] == 1
    assert result["evidence_assessment_failed"] is True
    assert "TimeoutError" in result["error"]
    assert "unverified answer" in result["answer"].lower()

@pytest.mark.asyncio
async def test_answer_generation_failure_returns_controlled_status():
    FakeLLM.assessments = [
        assessment(True, "The repository context answers the question.")
    ]

    original_ainvoke = FakeLLM.ainvoke

    async def fail_during_answer_generation(self, messages):
        if FakeLLM.assessment_index < len(FakeLLM.assessments):
            result = FakeLLM.assessments[FakeLLM.assessment_index]
            FakeLLM.assessment_index += 1
            return result

        raise TimeoutError("Simulated answer-generation timeout")

    FakeLLM.ainvoke = fail_during_answer_generation

    try:
        result = await build_graph(tools=[]).ainvoke(make_state())
    finally:
        FakeLLM.ainvoke = original_ainvoke

    assert result["status"] == "answer_generation_failed"
    assert "couldn't generate the final answer" in result["answer"].lower()
    assert "TimeoutError" in result["error"]