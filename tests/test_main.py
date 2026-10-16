
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