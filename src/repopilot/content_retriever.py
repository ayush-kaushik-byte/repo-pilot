from dataclasses import dataclass
from typing import Any

from repopilot.github_mcp import inspect_path


@dataclass
class RetrievedFile:
    """Result of attempting to retrieve one repository file."""

    path: str
    content: str = ""
    status: str = "pending"
    error: str | None = None
    truncated: bool = False


@dataclass
class ContentRetriever:
    """Retrieve selected source files through the existing GitHub MCP tools."""

    max_files: int = 10
    max_file_chars: int = 30_000
    max_total_chars: int = 100_000

    def __post_init__(self) -> None:
        if self.max_files < 1:
            raise ValueError("max_files must be at least 1")

        if self.max_file_chars < 1 or self.max_total_chars < 1:
            raise ValueError("Content limits must be positive")

    async def retrieve(
        self,
        tools: list,
        owner: str,
        repo: str,
        paths: list[str],
    ) -> list[RetrievedFile]:
        """
        Retrieve files in input order.

        Individual failures are recorded without discarding other results.
        Content limits apply to successfully retrieved file contents.
        """

        unique_paths = list(dict.fromkeys(paths))

        if len(unique_paths) > self.max_files:
            raise ValueError(
                f"Requested {len(unique_paths)} files; "
                f"maximum allowed is {self.max_files}."
            )

        results: list[RetrievedFile] = []
        total_chars = 0

        for path in unique_paths:
            if not self._is_safe_relative_path(path):
                results.append(
                    RetrievedFile(
                        path=path,
                        status="failed",
                        error="Invalid repository-relative path.",
                    )
                )
                continue

            remaining = self.max_total_chars - total_chars

            if remaining <= 0:
                results.append(
                    RetrievedFile(
                        path=path,
                        status="skipped",
                        error="Total content limit reached.",
                    )
                )
                continue

            try:
                raw_result = await inspect_path(
                    tools=tools,
                    owner=owner,
                    repo=repo,
                    path=path,
                )

                content = self._to_text(raw_result)

                if not content.strip():
                    results.append(
                        RetrievedFile(
                            path=path,
                            status="failed",
                            error="No file content returned.",
                        )
                    )
                    continue

                allowed_chars = min(
                    self.max_file_chars,
                    remaining,
                )

                truncated = len(content) > allowed_chars
                content = content[:allowed_chars]

                results.append(
                    RetrievedFile(
                        path=path,
                        content=content,
                        status="success",
                        truncated=truncated,
                    )
                )

                total_chars += len(content)

            except Exception as exc:
                # Keep the error isolated to this file.
                results.append(
                    RetrievedFile(
                        path=path,
                        status="failed",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )

        return results

    @staticmethod
    def _is_safe_relative_path(path: str) -> bool:
        """Reject absolute paths and traversal outside the repository."""

        if not isinstance(path, str) or not path.strip():
            return False

        normalized = path.replace("\\", "/")

        if normalized.startswith("/"):
            return False

        if any(part == ".." for part in normalized.split("/")):
            return False

        if "\x00" in normalized:
            return False

        return True

    @staticmethod
    def _to_text(result: Any) -> str:
        """Convert common MCP response shapes into text."""

        if isinstance(result, str):
            return result

        if isinstance(result, bytes):
            return result.decode("utf-8", errors="replace")

        if isinstance(result, dict):
            # Support structured wrappers that contain file text.
            for key in ("content", "text", "file_content"):
                value = result.get(key)

                if isinstance(value, (str, bytes)):
                    return ContentRetriever._to_text(value)

            # JSON source files may themselves decode to dictionaries.
            import json
            return json.dumps(result, ensure_ascii=False, indent=2)

        raise TypeError(
            f"Unsupported MCP file response type: "
            f"{type(result).__name__}"
        )