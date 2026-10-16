import json
import os
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_mcp_adapters.tools import load_mcp_tools


load_dotenv()


def create_github_mcp_client() -> MultiServerMCPClient:
    """Create the GitHub MCP client configuration."""

    github_token = os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN")

    if not github_token:
        raise RuntimeError(
            "GITHUB_PERSONAL_ACCESS_TOKEN is not set in .env"
        )

    return MultiServerMCPClient(
        {
            "github": {
                "transport": "stdio",
                "command": "docker",
                "args": [
                    "run",
                    "--rm",
                    "-i",
                    "--env",
                    f"GITHUB_PERSONAL_ACCESS_TOKEN={github_token}",
                    "-e",
                    "GITHUB_READ_ONLY=1",
                    "-e",
                    "GITHUB_TOOLSETS=repos",
                    "ghcr.io/github/github-mcp-server",
                ],
            }
        }
    )


@asynccontextmanager
async def github_mcp_session() -> AsyncIterator[list]:
    """
    Start one persistent GitHub MCP session.

    All MCP tool calls made inside this context use the same
    MCP server process/session.
    """

    client = create_github_mcp_client()

    async with client.session("github") as session:
        tools = await load_mcp_tools(session)

        print(f"Discovered {len(tools)} GitHub MCP tools.")

        yield tools


def normalize_mcp_result(result: Any) -> Any:
    if not isinstance(result, list):
        return result

    text_items = [
        item.get("text", "")
        for item in result
        if isinstance(item, dict) and item.get("type") == "text"
    ]

    if not text_items:
        return result

    # Directory/list responses are returned as JSON text.
    for text in text_items:
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            continue

    # File responses may contain a status message followed
    # by the actual file contents.
    meaningful_text = [
        text
        for text in text_items
        if not text.startswith("successfully downloaded text file")
    ]

    if meaningful_text:
        return "\n".join(meaningful_text)

    return "\n".join(text_items)


async def inspect_path(
    tools: list,
    owner: str,
    repo: str,
    path: str = "",
):
    """
    Inspect either a directory or a file in a GitHub repository.
    """

    tool_map = {tool.name: tool for tool in tools}

    get_file_contents = tool_map.get("get_file_contents")

    if get_file_contents is None:
        raise RuntimeError(
            "GitHub MCP tool 'get_file_contents' was not found."
        )

    result = await get_file_contents.ainvoke(
        {
            "owner": owner,
            "repo": repo,
            "path": path,
        }
    )

    return normalize_mcp_result(result)