from dataclasses import dataclass, field

from repopilot.content_retriever import RetrievedFile


@dataclass
class RepositoryContext:
    """Structured evidence prepared for the reasoning agent."""

    text: str = ""
    included_paths: list[str] = field(default_factory=list)
    failed_paths: list[str] = field(default_factory=list)
    skipped_paths: list[str] = field(default_factory=list)
    truncated_paths: list[str] = field(default_factory=list)

    @property
    def total_chars(self) -> int:
        return len(self.text)


@dataclass
class RepositoryContextBuilder:
    """Build a bounded text context from retrieved repository files."""

    max_total_chars: int = 50_000

    def __post_init__(self) -> None:
        if self.max_total_chars < 1:
            raise ValueError("max_total_chars must be at least 1")

    def build(
        self,
        files: list[RetrievedFile],
    ) -> RepositoryContext:
        context = RepositoryContext()
        sections: list[str] = []
        used_chars = 0

        for file in files:
            if file.status == "failed":
                context.failed_paths.append(file.path)
                continue

            if file.status == "skipped":
                context.skipped_paths.append(file.path)
                continue

            if file.status != "success" or not file.content.strip():
                context.skipped_paths.append(file.path)
                continue

            section = (
                f"--- FILE: {file.path} ---\n"
                f"{file.content}\n"
            )

            remaining = self.max_total_chars - used_chars

            if len(section) > remaining:
                context.skipped_paths.append(file.path)
                continue

            sections.append(section)
            used_chars += len(section)
            context.included_paths.append(file.path)

            if file.truncated:
                context.truncated_paths.append(file.path)

        context.text = "\n".join(sections)
        return context