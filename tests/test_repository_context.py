import pytest

from repopilot.content_retriever import RetrievedFile
from repopilot.repository_context import RepositoryContextBuilder


def test_builds_context_with_file_paths():
    files = [
        RetrievedFile(
            path="src/app.py",
            content="def hello(): return 'hi'",
            status="success",
        ),
        RetrievedFile(
            path="tests/test_app.py",
            content="def test_hello(): assert True",
            status="success",
        ),
    ]

    context = RepositoryContextBuilder().build(files)

    assert context.included_paths == [
        "src/app.py",
        "tests/test_app.py",
    ]
    assert "--- FILE: src/app.py ---" in context.text
    assert "def hello(): return 'hi'" in context.text
    assert "--- FILE: tests/test_app.py ---" in context.text
    assert context.total_chars == len(context.text)


def test_failed_files_are_reported_but_not_included():
    files = [
        RetrievedFile(
            path="src/missing.py",
            status="failed",
            error="Not found",
        ),
        RetrievedFile(
            path="src/working.py",
            content="print('hello')",
            status="success",
        ),
    ]

    context = RepositoryContextBuilder().build(files)

    assert context.failed_paths == ["src/missing.py"]
    assert context.included_paths == ["src/working.py"]
    assert "Not found" not in context.text


def test_skipped_files_are_reported():
    files = [
        RetrievedFile(
            path="src/large.py",
            status="skipped",
            error="Total content limit reached.",
        )
    ]

    context = RepositoryContextBuilder().build(files)

    assert context.skipped_paths == ["src/large.py"]
    assert context.text == ""


def test_truncated_files_are_marked():
    files = [
        RetrievedFile(
            path="src/partial.py",
            content="print('partial')",
            status="success",
            truncated=True,
        )
    ]

    context = RepositoryContextBuilder().build(files)

    assert context.included_paths == ["src/partial.py"]
    assert context.truncated_paths == ["src/partial.py"]


def test_enforces_total_context_character_limit():
    files = [
        RetrievedFile(
            path="src/first.py",
            content="x = 1",
            status="success",
        ),
        RetrievedFile(
            path="src/second.py",
            content="y = 2",
            status="success",
        ),
    ]

    # Exactly enough room for the first rendered file section.
    first_only = RepositoryContextBuilder().build(files[:1])
    context = RepositoryContextBuilder(
        max_total_chars=len(first_only.text)
    ).build(files)

    assert context.included_paths == ["src/first.py"]
    assert context.skipped_paths == ["src/second.py"]
    assert context.total_chars <= len(first_only.text)


def test_skips_file_that_does_not_fit_and_keeps_later_small_file():
    files = [
        RetrievedFile(
            path="src/large.py",
            content="x" * 1000,
            status="success",
        ),
        RetrievedFile(
            path="a.py",
            content="x",
            status="success",
        ),
    ]

    context = RepositoryContextBuilder(max_total_chars=100).build(files)

    assert context.skipped_paths == ["src/large.py"]
    assert context.included_paths == ["a.py"]


def test_rejects_non_positive_context_limit():
    with pytest.raises(ValueError):
        RepositoryContextBuilder(max_total_chars=0)


def test_empty_content_is_not_added_to_context():
    files = [
        RetrievedFile(
            path="src/empty.py",
            content="   ",
            status="success",
        )
    ]

    context = RepositoryContextBuilder().build(files)

    assert context.included_paths == []
    assert context.skipped_paths == ["src/empty.py"]
    assert context.text == ""