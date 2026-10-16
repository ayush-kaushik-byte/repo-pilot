import asyncio

from repopilot.github_mcp import github_mcp_session
from repopilot.repository_discovery import RepositoryDiscoverer
from repopilot.repository_analyzer import RepositoryAnalyzer
from repopilot.candidate_generator import CandidateGenerator


OWNER = "ayush-kaushik-byte"
REPOSITORY = "Django_Repository"

USER_QUERY = "Where is the application logic implemented?"


async def main():

    # ---------------------------------------------------------
    # 1. Start GitHub MCP session
    # ---------------------------------------------------------

    async with github_mcp_session() as tools:

        # -----------------------------------------------------
        # 2. Discover repository structure
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # 3. Analyze repository
        # -----------------------------------------------------

        analyzer = RepositoryAnalyzer()

        analysis = analyzer.analyze(snapshot)

        print("\n=== Repository Analysis ===")

        print("\nLanguages:")
        for language, count in analysis.languages.items():
            print(f"  {language}: {count}")

        print("\nFrameworks:")
        for framework in analysis.frameworks:
            print(f"  {framework}")

        print("\nDependency files:")
        for path in analysis.dependency_files:
            print(f"  {path}")

        print("\nEntry points:")
        for path in analysis.entry_points:
            print(f"  {path}")

        print("\nTest files:")
        for path in analysis.test_files:
            print(f"  {path}")

        # -----------------------------------------------------
        # 4. Generate deterministic candidates
        # -----------------------------------------------------

        generator = CandidateGenerator(
            max_candidates=20,
        )

        candidates = generator.generate(
            snapshot=snapshot,
            analysis=analysis,
            query=USER_QUERY,
        )

        # -----------------------------------------------------
        # 5. Display candidates
        # -----------------------------------------------------

        print("\n=== Candidate Files ===")

        for index, candidate in enumerate(candidates, start=1):

            print(
                f"\n{index}. "
                f"[score={candidate.score:.1f}] "
                f"{candidate.path}"
            )

            for reason in candidate.reasons:
                print(f"     - {reason}")


if __name__ == "__main__":
    asyncio.run(main())