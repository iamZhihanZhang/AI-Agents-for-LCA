# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - LLM interface for the OpenAI API

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import os
from agentic_lci.llm_adapters.interface import LLM, T, Literal, encode_image

from openai import OpenAI

client = OpenAI()

from dotenv import load_dotenv

load_dotenv()

assert os.environ.get(
    "OPENAI_API_KEY"
), "Please add an OPENAI_API_KEY to the .env file."


class OpenAILLM(LLM[dict]):
    def __init__(self, model: str = "gpt-5"):
        self.model = model

    def create_message(
        self,
        role: Literal["system", "user", "assistant"],
        prompt: str | None = None,
        image_url: str | None = None,
    ) -> dict:
        content = []
        if prompt:
            content.append({"type": "input_text", "text": prompt})
        if image_url:
            if not image_url.startswith("http"):
                image_url = f"data:image/jpeg;base64,{encode_image(image_url)}"
            content.append({"type": "input_image", "image_url": image_url})
        return {"role": role, "content": content}

    def unstructured_output(
        self, messages: list[dict], use_mcp: bool = False
    ) -> tuple[str, int]:
        from agentic_lci.mcp.function_call import TOOLS, call_tool

        response = client.responses.create(
            model=self.model,
            input=messages,  # type: ignore
            tools=[
                {"type": "web_search"},  # built-in web search
                *([t["spec"] for t in TOOLS.values()] if use_mcp else []),  # type: ignore
            ],
            tool_choice="auto",
        )
        messages += response.output  # type: ignore
        for item in response.output:  # type: ignore
            if item.type == "function_call":
                messages.append(call_tool(item.name, item.call_id, item.arguments))
        total_tokens = response.usage.total_tokens  # type: ignore
        return response.output_text, total_tokens

    def structured_output(
        self, messages: list[dict], output_type: type[T]
    ) -> tuple[T, int]:
        response = client.responses.parse(
            model=self.model,
            input=messages,  # type: ignore
            tools=[
                {"type": "web_search"},  # built-in web search
            ],
            text_format=output_type,
        )
        messages += response.output  # type: ignore
        total_tokens = response.usage.total_tokens  # type: ignore
        return response.output_parsed, total_tokens  # type: ignore


if __name__ == "__main__":
    # example/test usage
    from pydantic import BaseModel

    class SimpleStructure(BaseModel):
        footprint: int
        unit: str

    llm: LLM = OpenAILLM()
    messages = [
        llm.create_message(
            "user",
            "use the lookup_db_info tool on the iPhone 13 to estimate its carbon footprint",
        )
    ]
    print(llm.unstructured_output(messages, True))
    print(llm.structured_output(messages, SimpleStructure))
