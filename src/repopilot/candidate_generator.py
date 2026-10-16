from dataclasses import dataclass, field
from pathlib import PurePosixPath

from repopilot.repository_discovery import RepositorySnapshot
from repopilot.repository_analyzer import RepositoryAnalysis


@dataclass
class CandidateFile:
    path: str
    score: float
    reasons: list[str] = field(default_factory=list)


class CandidateGenerator:

    SOURCE_EXTENSIONS = {
        ".py",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".java",
        ".go",
        ".rs",
        ".cpp",
        ".c",
        ".cs",
        ".rb",
        ".php",
        ".kt",
        ".swift",
    }

    CONFIG_FILENAMES = {
        ".env.example",
        "dockerfile",
        "docker-compose.yml",
        "docker-compose.yaml",
        "makefile",
        "jenkinsfile",
        "config.yaml",
        "config.yml",
        "settings.py",
        "application.yml",
        "application.yaml",
    }

    # Files that are structurally important but usually
    # shouldn't dominate the candidate set.
    LOW_VALUE_SOURCE_NAMES = {
        "__init__.py",
        "asgi.py",
        "wsgi.py",
    }

    def __init__(
        self,
        max_candidates: int = 20,
    ):
        self.max_candidates = max_candidates

    def generate(
        self,
        snapshot: RepositorySnapshot,
        analysis: RepositoryAnalysis,
        query: str | None = None,
    ) -> list[CandidateFile]:

        candidates: list[CandidateFile] = []

        important_paths = set(
            analysis.entry_points
            + analysis.dependency_files
            + analysis.test_files
        )

        for file in snapshot.files:

            path = PurePosixPath(file.path)
            suffix = path.suffix.lower()
            name = file.name.lower()

            score = 0.0
            reasons: list[str] = []

            # -------------------------------------------------
            # 1. Repository entry point
            # -------------------------------------------------

            if file.path in analysis.entry_points:
                score += 5
                reasons.append("repository entry point")

            # -------------------------------------------------
            # 2. Dependency manifest
            # -------------------------------------------------

            if file.path in analysis.dependency_files:
                score += 4
                reasons.append("dependency manifest")

            # -------------------------------------------------
            # 3. Test file
            # -------------------------------------------------

            if file.path in analysis.test_files:
                score += 3
                reasons.append("test file")

            # -------------------------------------------------
            # 4. Configuration / deployment file
            # -------------------------------------------------

            if name in self.CONFIG_FILENAMES:
                score += 3
                reasons.append("configuration/deployment file")

            # -------------------------------------------------
            # 5. Source-code signal
            #
            # Source code is useful, but we don't give every
            # source file a strong score anymore.
            # -------------------------------------------------

            is_source_file = suffix in self.SOURCE_EXTENSIONS

            if is_source_file:
                # Source files get a small baseline signal.
                score += 0.5
                reasons.append("source code")

            # -------------------------------------------------
            # 6. Relationship to important files
            #
            # Generic structural relationship:
            # files in the same directory as important files
            # receive additional relevance.
            # -------------------------------------------------

            for important_path in important_paths:

                important_parent = PurePosixPath(
                    important_path
                ).parent

                if path.parent == important_parent:
                    score += 2
                    reasons.append(
                        "located near an important repository file"
                    )
                    break

            # -------------------------------------------------
            # 7. Query/path lexical relevance
            #
            # This is intentionally only a ranking signal.
            # Semantic understanding belongs to the LLM.
            # -------------------------------------------------

            if query:

                query_terms = self._tokenize(query)

                path_tokens = self._tokenize(file.path)

                matched_terms = [
                    term
                    for term in query_terms
                    if term in path_tokens
                ]

                if matched_terms:
                    score += min(
                        3.0,
                        1.0 * len(matched_terms),
                    )

                    reasons.append(
                        "path matches user request: "
                        + ", ".join(matched_terms)
                    )

            # -------------------------------------------------
            # 8. De-prioritize low-value structural files
            # -------------------------------------------------

            if name in self.LOW_VALUE_SOURCE_NAMES:

                score -= 1

                reasons.append(
                    "low-value structural source file"
                )

            # -------------------------------------------------
            # Don't include files with no meaningful signal.
            #
            # The source-code baseline alone is intentionally
            # not enough.
            # -------------------------------------------------

            meaningful_signal = (
                file.path in important_paths
                or name in self.CONFIG_FILENAMES
                or score >= 2
            )

            if not meaningful_signal:
                continue

            candidates.append(
                CandidateFile(
                    path=file.path,
                    score=score,
                    reasons=reasons,
                )
            )

        # Highest relevance first.
        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.path,
            )
        )

        return candidates[: self.max_candidates]

    @staticmethod
    def _tokenize(text: str) -> list[str]:

        normalized = (
            text.lower()
            .replace("/", " ")
            .replace("_", " ")
            .replace("-", " ")
            .replace(".", " ")
        )

        return [
            token
            for token in normalized.split()
            if len(token) >= 3
        ]