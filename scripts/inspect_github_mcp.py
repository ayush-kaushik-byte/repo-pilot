import asyncio

from repopilot.github_mcp import (
    github_mcp_session,
    inspect_path,
)


OWNER = "ayush-kaushik-byte"
REPOSITORY = "Django_Repository"


async def main():
    async with github_mcp_session() as tools:

        print("\n=== Repository Root ===")

        root = await inspect_path(
            tools=tools,
            owner=OWNER,
            repo=REPOSITORY,
            path="",
        )

        print(root)

        print("\n=== Inspecting 'mysite/' ===")

        mysite = await inspect_path(
            tools=tools,
            owner=OWNER,
            repo=REPOSITORY,
            path="mysite",
        )

        print(mysite)

        print("\n=== Inspecting 'mysite/manage.py' ===")

        manage_py = await inspect_path(
            tools=tools,
            owner=OWNER,
            repo=REPOSITORY,
            path="mysite/manage.py",
        )

        print("Value:", manage_py)


if __name__ == "__main__":
    asyncio.run(main())