
from types import SimpleNamespace

from repopilot.main import format_progress_event


def test_formats_basic_node_progress():
    messages = format_progress_event({
        "discover_repository": {"snapshot": {}},
        "analyze_repository": {"analysis": {}},
    })

    assert any("Discovering repository" in m for m in messages)
    assert any("Analyzing repository" in m for m in messages)


def test_formats_retrieval_counts():
    files = [
        SimpleNamespace(status="success"),
        SimpleNamespace(status="success"),
        SimpleNamespace(status="failed"),
    ]

    messages = format_progress_event({
        "retrieve_files": {"retrieved_files": files}
    })

    assert any("2 successful" in m and "1 failed" in m for m in messages)


def test_formats_context_summary():
    context = SimpleNamespace(
        included_paths=["src/app.py", "src/routes.py"],
        total_chars=1250,
    )

    messages = format_progress_event({
        "build_context": {"context": context}
    })

    assert any("2 files" in m and "1250 characters" in m for m in messages)


def test_formats_evidence_recovery():
    messages = format_progress_event({
        "check_evidence": {
            "evidence_sufficient": False,
            "evidence_assessment_failed": False,
        },
        "recover_files": {},
    })

    assert any("insufficient" in m.lower() for m in messages)
    assert any("recovery" in m.lower() for m in messages)


def test_formats_assessment_failure():
    messages = format_progress_event({
        "check_evidence": {
            "evidence_sufficient": False,
            "evidence_assessment_failed": True,
        }
    })

    assert any("assessment" in m.lower() and "error" in m.lower()
               for m in messages)


def test_formats_evidence_reason():
    messages = format_progress_event({
        "check_evidence": {
            "evidence_sufficient": False,
            "evidence_assessment_failed": False,
            "evidence_reason": "The request flow is not fully represented.",
        }
    })

    assert any(
        "The request flow is not fully represented." in message
        for message in messages
    )


def test_formats_recovery_attempt_count():
    messages = format_progress_event({
        "recover_files": {"recovery_attempts": 1}
    })

    assert any("Recovery attempt count: 1" in message for message in messages)


def test_formats_terminal_workflow_status():
    messages = format_progress_event({
        "finish_insufficient_evidence": {
            "status": "insufficient_evidence"
        }
    })

    assert any(
        "Workflow stopped: insufficient_evidence" in message
        for message in messages
    )


def test_does_not_report_missing_terminal_status():
    messages = format_progress_event({
        "finish_no_candidates": {}
    })

    assert not any("Workflow stopped:" in message for message in messages)
