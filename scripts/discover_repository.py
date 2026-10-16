import asyncio

from repopilot.github_mcp import github_mcp_session
from repopilot.repository_discovery import RepositoryDiscoverer


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

        print("\n=== Repository ===")
        print(f"{snapshot.owner}/{snapshot.repository}")

        print("\n=== Statistics ===")
        print(f"Directories: {snapshot.directory_count}")
        print(f"Files:       {snapshot.file_count}")

        print("\n=== Directories ===")

        for directory in snapshot.directories:
            print(f"[DIR ] {directory.path}")

        print("\n=== Files ===")

        for file in snapshot.files:
            print(
                f"[FILE] {file.path} "
                f"({file.size} bytes)"
            )


if __name__ == "__main__":
    asyncio.run(main())