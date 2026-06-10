# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Test interface for the MCP server

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import asyncio
from fastmcp.client import Client
from agentic_lci.mcp.server import mcp


async def main():
    async with Client(mcp) as client:  # type: ignore
        print(await client.list_tools())

        print(
            await client.call_tool(
                "lookup_db_info", dict(product_name="iphone 13 mini")
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
