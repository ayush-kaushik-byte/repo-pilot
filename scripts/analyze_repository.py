import asyncio

from repopilot.github_mcp import github_mcp_session
from repopilot.repository_discovery import RepositoryDiscoverer
from repopilot.repository_analyzer import RepositoryAnalyzer


OWNER = "ayush-kaushik-byte"
REPOSITORY = "Django_Repository"


async def main():

    async with github_mcp_session() as tools:

        discoverer = RepositoryDiscoverer(
            tools=tools,
            owner=OWNER,
            repository=REPOSITORY,
        )

        snapshot = await discoverer.discover()

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
        
        print("\nEvidence:")

        for evidence in analysis.evidence:
            print(
                f"  [{evidence.signal}] "
                f"{evidence.source} -> "
                f"{evidence.description}"
            )


if __name__ == "__main__":
    asyncio.run(main())
