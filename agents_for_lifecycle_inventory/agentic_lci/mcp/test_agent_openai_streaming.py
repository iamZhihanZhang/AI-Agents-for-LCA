# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Single agent streaming test for the tools

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import base64
import asyncio
from openai import OpenAI
from agents import Agent, Runner
from agents.mcp import MCPServerStreamableHttp
from openai.types.responses import ResponseTextDeltaEvent

from dotenv import load_dotenv

load_dotenv()

client = OpenAI()


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


async def main():
    async with MCPServerStreamableHttp(
        params={"url": "http://127.0.0.1:8000/mcp"}
    ) as server:
        # print(await server.list_tools())

        agent = Agent(
            name="LCA Specialist",
            instructions="""
            You are an experienced Life Cycle Assessment (LCA) Specialist.
            You will use the tools to estimate the CO2eq in kg of the products requested by the user.
            """,
            mcp_servers=[server],
        )

        # prompt = "What is the carbon footprint of an iphone 13 mini?"
        prompt = "What is the carbon footprint of the phone in this image?"
        image_path = "data/iPhone13Internal.png"
        result = Runner.run_streamed(
            agent,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {
                            "type": "input_image",
                            "image_url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                        },
                    ],
                }  # type: ignore
            ],
        )

        async for event in result.stream_events():
            if event.type == "raw_response_event" and isinstance(
                event.data, ResponseTextDeltaEvent
            ):
                print(event.data.delta, end="", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
