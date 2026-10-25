import asyncio

from repopilot.github_mcp import github_mcp_session
from repopilot.repository_discovery import RepositoryDiscoverer
from repopilot.repository_analyzer import RepositoryAnalyzer
from repopilot.candidate_generator import CandidateGenerator
from repopilot.file_selector import FileSelector


OWNER = "ayush-kaushik-byte"
REPOSITORY = "Django_Repository"
USER_QUERY = "Where is the application logic implemented?"


async def main():

    async with github_mcp_session() as tools:

        # ---------------------------------------------------------
        # 1. Discover repository
        # ---------------------------------------------------------
        discoverer = RepositoryDiscoverer(
            tools=tools,
            owner=OWNER,
            repository=REPOSITORY,
        )

        snapshot = await discoverer.discover()

        print("\n=== Repository Snapshot ===")
        print(f"Files: {snapshot.file_count}")
        print(f"Directories: {snapshot.directory_count}")
        print(f"Truncated: {snapshot.truncated}")

        # ---------------------------------------------------------
        # 2. Analyze repository
        # ---------------------------------------------------------
        analyzer = RepositoryAnalyzer()
        analysis = analyzer.analyze(snapshot)

        print("\n=== Repository Analysis ===")
        print(f"Languages: {analysis.languages}")
        print(f"Frameworks: {analysis.frameworks}")
        print(f"Dependency files: {analysis.dependency_files}")
        print(f"Entry points: {analysis.entry_points}")
        print(f"Test files: {analysis.test_files}")

        # ---------------------------------------------------------
        # 3. Deterministic candidate generation
        # ---------------------------------------------------------
        generator = CandidateGenerator(
            max_candidates=20
        )

        candidates = generator.generate(
            snapshot=snapshot,
            analysis=analysis,
            query=USER_QUERY,
        )

        print("\n=== Candidate Files ===")

        for index, candidate in enumerate(candidates, start=1):
            print(
                f"{index}. "
                f"[score={candidate.score:.1f}] "
                f"{candidate.path}"
            )

            for reason in candidate.reasons:
                print(f"     - {reason}")

        # ---------------------------------------------------------
        # 4. LLM file selection
        # ---------------------------------------------------------
        selector = FileSelector(
            model_name="llama3.2:3b",
            temperature=0.0,
        )

        selection = await selector.select(
            query=USER_QUERY,
            candidates=candidates,
        )

        selection = selector.validate_selection(
            selection=selection,
            candidates=candidates,
        )

        # ---------------------------------------------------------
        # 5. Display LLM selection
        # ---------------------------------------------------------
        print("\n=== LLM File Selection ===")

        for index, selected in enumerate(
            selection.selected_files,
            start=1,
        ):
            print(f"\n{index}. {selected.path}")
            # print(f"   relevance: {selected.relevance_score:.2f}")
            print(f"   reason: {selected.reason}")


if __name__ == "__main__":
    asyncio.run(main())