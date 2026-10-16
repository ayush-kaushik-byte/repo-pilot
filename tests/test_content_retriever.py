import pytest

import repopilot.content_retriever as module
from repopilot.content_retriever import ContentRetriever


@pytest.mark.asyncio
async def test_retrieves_multiple_files(monkeypatch):
    async def fake_inspect_path(tools, owner, repo, path):
        return f"contents of {path}"

    monkeypatch.setattr(module, "inspect_path", fake_inspect_path)

    retriever = ContentRetriever()

    results = await retriever.retrieve(
        tools=[],
        owner="example",
        repo="project",
        paths=["src/a.py", "src/b.py"],
    )

    assert [r.status for r in results] == ["success", "success"]
    assert results[0].content == "contents of src/a.py"
    assert results[1].path == "src/b.py"


@pytest.mark.asyncio
async def test_duplicate_paths_are_retrieved_once(monkeypatch):
    calls = []

    async def fake_inspect_path(tools, owner, repo, path):
        calls.append(path)
        return "print('hello')"

    monkeypatch.setattr(module, "inspect_path", fake_inspect_path)

    results = await ContentRetriever().retrieve(
        tools=[],
        owner="example",
        repo="project",
        paths=["src/a.py", "src/a.py"],
    )

    assert len(results) == 1
    assert calls == ["src/a.py"]


@pytest.mark.asyncio
async def test_failure_does_not_discard_other_files(monkeypatch):
    async def fake_inspect_path(tools, owner, repo, path):
        if path == "missing.py":
            raise RuntimeError("Not found")
        return "valid source"

    monkeypatch.setattr(module, "inspect_path", fake_inspect_path)

    results = await ContentRetriever().retrieve(
        tools=[],
        owner="example",
        repo="project",
        paths=["missing.py", "valid.py"],
    )

    assert results[0].status == "failed"
    assert results[1].status == "success"
    assert results[1].content == "valid source"


@pytest.mark.asyncio
async def test_per_file_character_limit(monkeypatch):
    async def fake_inspect_path(tools, owner, repo, path):
        return "abcdefghij"

    monkeypatch.setattr(module, "inspect_path", fake_inspect_path)

    results = await ContentRetriever(
        max_file_chars=5,
    ).retrieve(
        tools=[],
        owner="example",
        repo="project",
        paths=["large.py"],
    )

    assert results[0].content == "abcde"
    assert results[0].truncated is True


@pytest.mark.asyncio
async def test_total_character_limit(monkeypatch):
    async def fake_inspect_path(tools, owner, repo, path):
        return "abcdefgh"

    monkeypatch.setattr(module, "inspect_path", fake_inspect_path)

    results = await ContentRetriever(
        max_file_chars=10,
        max_total_chars=10,
    ).retrieve(
        tools=[],
        owner="example",
        repo="project",
        paths=["a.py", "b.py"],
    )

    assert results[0].status == "success"
    assert results[0].content == "abcdefgh"
    assert results[1].status == "success"
    assert results[1].content == "ab"
    assert results[1].truncated is True
    assert sum(len(r.content) for r in results) <= 10


@pytest.mark.asyncio
async def test_rejects_path_traversal(monkeypatch):
    async def fake_inspect_path(tools, owner, repo, path):
        pytest.fail("Unsafe paths must not reach MCP")

    monkeypatch.setattr(module, "inspect_path", fake_inspect_path)

    results = await ContentRetriever().retrieve(
        tools=[],
        owner="example",
        repo="project",
        paths=["../secrets.txt"],
    )

    assert results[0].status == "failed"


@pytest.mark.asyncio
async def test_rejects_too_many_files():
    retriever = ContentRetriever(max_files=1)

    with pytest.raises(ValueError, match="maximum allowed"):
        await retriever.retrieve(
            tools=[],
            owner="example",
            repo="project",
            paths=["a.py", "b.py"],
        )


@pytest.mark.asyncio
async def test_empty_content_is_a_failure(monkeypatch):
    async def fake_inspect_path(tools, owner, repo, path):
        return "   "

    monkeypatch.setattr(module, "inspect_path", fake_inspect_path)

    results = await ContentRetriever().retrieve(
        tools=[],
        owner="example",
        repo="project",
        paths=["empty.py"],
    )

    assert results[0].status == "failed"
    assert results[0].error == "No file content returned."