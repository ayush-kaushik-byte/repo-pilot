from dataclasses import dataclass, field
from pathlib import PurePosixPath

from repopilot.repository_discovery import RepositorySnapshot


@dataclass
class AnalysisEvidence:
    signal: str
    source: str
    description: str


@dataclass
class RepositoryAnalysis:
    languages: dict[str, int] = field(default_factory=dict)
    frameworks: list[str] = field(default_factory=list)
    dependency_files: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    test_files: list[str] = field(default_factory=list)
    evidence: list[AnalysisEvidence] = field(default_factory=list)


class RepositoryAnalyzer:

    LANGUAGE_MAP = {
        ".py": "Python",
        ".js": "JavaScript",
        ".jsx": "JavaScript",
        ".ts": "TypeScript",
        ".tsx": "TypeScript",
        ".java": "Java",
        ".go": "Go",
        ".rs": "Rust",
        ".cpp": "C++",
        ".c": "C",
        ".cs": "C#",
        ".rb": "Ruby",
        ".php": "PHP",
        ".kt": "Kotlin",
        ".swift": "Swift",
    }

    DEPENDENCY_FILES = {
        "requirements.txt",
        "pyproject.toml",
        "Pipfile",
        "poetry.lock",
        "package.json",
        "package-lock.json",
        "yarn.lock",
        "pom.xml",
        "build.gradle",
        "go.mod",
        "Cargo.toml",
        "Gemfile",
        "composer.json",
    }

    ENTRY_POINT_NAMES = {
        "main.py",
        "app.py",
        "server.py",
        "manage.py",
        "index.js",
        "index.ts",
        "main.go",
        "main.java",
    }

    def analyze(
        self,
        snapshot: RepositorySnapshot,
    ) -> RepositoryAnalysis:

        analysis = RepositoryAnalysis()

        self._detect_languages(snapshot, analysis)
        self._detect_dependency_files(snapshot, analysis)
        self._detect_entry_points(snapshot, analysis)
        self._detect_tests(snapshot, analysis)
        self._detect_frameworks(snapshot, analysis)

        return analysis

    def _detect_languages(
        self,
        snapshot: RepositorySnapshot,
        analysis: RepositoryAnalysis,
    ) -> None:

        for file in snapshot.files:

            suffix = PurePosixPath(file.name).suffix.lower()

            language = self.LANGUAGE_MAP.get(suffix)

            if language:
                analysis.languages[language] = (
                    analysis.languages.get(language, 0) + 1
                )

                analysis.evidence.append(
                    AnalysisEvidence(
                        signal="language",
                        source=file.path,
                        description=f"File extension indicates {language}.",
                    )
                )

    def _detect_dependency_files(
        self,
        snapshot: RepositorySnapshot,
        analysis: RepositoryAnalysis,
    ) -> None:

        for file in snapshot.files:

            if file.name in self.DEPENDENCY_FILES:
                analysis.dependency_files.append(file.path)

    def _detect_entry_points(
        self,
        snapshot: RepositorySnapshot,
        analysis: RepositoryAnalysis,
    ) -> None:

        for file in snapshot.files:

            if file.name in self.ENTRY_POINT_NAMES:
                analysis.entry_points.append(file.path)

    def _detect_tests(
        self,
        snapshot: RepositorySnapshot,
        analysis: RepositoryAnalysis,
    ) -> None:

        for file in snapshot.files:

            name = file.name.lower()
            path = file.path.lower()

            if (
                name.startswith("test_")
                or name.endswith("_test.py")
                or "/tests/" in f"/{path}/"
                or name == "tests.py"
            ):
                analysis.test_files.append(file.path)

    def _detect_frameworks(
        self,
        snapshot: RepositorySnapshot,
        analysis: RepositoryAnalysis,
    ) -> None:

        paths = {file.path for file in snapshot.files}
        names = {file.name for file in snapshot.files}

        # Python / Django
        if "manage.py" in names:
            analysis.frameworks.append("Django")

            analysis.evidence.append(
                AnalysisEvidence(
                    signal="framework",
                    source="manage.py",
                    description="Repository contains a Django management entry point.",
                )
            )

        # Python / FastAPI
        if any(
            path.endswith("requirements.txt")
            for path in paths
        ):
            # Dependency contents can be inspected later.
            pass

        # Node / React
        if "package.json" in names:
            analysis.frameworks.append("Node.js")

            analysis.evidence.append(
                AnalysisEvidence(
                    signal="runtime",
                    source="package.json",
                    description="Repository contains an npm package manifest.",
                )
            )
