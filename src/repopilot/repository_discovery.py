from dataclasses import dataclass, field

from repopilot.github_mcp import inspect_path


@dataclass
class RepositoryFile:
    path: str
    name: str
    size: int
    sha: str


@dataclass
class RepositoryDirectory:
    path: str
    name: str
    sha: str


@dataclass
class RepositorySnapshot:
    owner: str
    repository: str

    files: list[RepositoryFile] = field(default_factory=list)
    directories: list[RepositoryDirectory] = field(default_factory=list)

    truncated: bool = False

    @property
    def file_count(self) -> int:
        return len(self.files)

    @property
    def directory_count(self) -> int:
        return len(self.directories)


class RepositoryDiscoverer:
    """
    Discovers the structure of a GitHub repository through MCP.

    The discoverer is repository-agnostic and does not assume
    anything about the programming language or framework.
    """

    DEFAULT_IGNORED_DIRECTORIES = {
        ".git",
        ".github",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "env",
        ".tox",
        ".mypy_cache",
        ".pytest_cache",
    }

    def __init__(
        self,
        tools: list,
        owner: str,
        repository: str,
        max_depth: int = 10,
        max_files: int = 1000,
        max_directories: int = 500,
        ignored_directories: set[str] | None = None,
    ):
        self.tools = tools
        self.owner = owner
        self.repository = repository

        self.max_depth = max_depth
        self.max_files = max_files
        self.max_directories = max_directories

        self.ignored_directories = (
            ignored_directories
            if ignored_directories is not None
            else self.DEFAULT_IGNORED_DIRECTORIES
        )

    async def discover(self) -> RepositorySnapshot:
        snapshot = RepositorySnapshot(
            owner=self.owner,
            repository=self.repository,
        )

        await self._discover_directory(
            path="",
            depth=0,
            snapshot=snapshot,
        )

        return snapshot

    async def _discover_directory(
        self,
        path: str,
        depth: int,
        snapshot: RepositorySnapshot,
    ) -> None:

        if depth > self.max_depth:
            snapshot.truncated = True
            return

        if snapshot.file_count >= self.max_files:
            snapshot.truncated = True
            return

        if snapshot.directory_count >= self.max_directories:
            snapshot.truncated = True
            return

        result = await inspect_path(
            tools=self.tools,
            owner=self.owner,
            repo=self.repository,
            path=path,
        )

        if not isinstance(result, list):
            raise RuntimeError(
                f"Expected a directory listing for path {path!r}, "
                f"received {type(result).__name__}: {str(result)[:300]}"
            )

        for item in result:

            if not isinstance(item, dict):
                continue

            item_type = item.get("type")
            item_path = item.get("path", "")
            item_name = item.get("name", "")

            if item_type == "dir":

                if item_name in self.ignored_directories:
                    continue

                if snapshot.directory_count >= self.max_directories:
                    snapshot.truncated = True
                    return

                snapshot.directories.append(
                    RepositoryDirectory(
                        path=item_path,
                        name=item_name,
                        sha=item.get("sha", ""),
                    )
                )

                await self._discover_directory(
                    path=item_path,
                    depth=depth + 1,
                    snapshot=snapshot,
                )

            elif item_type == "file":

                if snapshot.file_count >= self.max_files:
                    snapshot.truncated = True
                    return

                snapshot.files.append(
                    RepositoryFile(
                        path=item_path,
                        name=item_name,
                        size=item.get("size", 0),
                        sha=item.get("sha", ""),
                    )
                )