# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - LLM interface for the Gemini API

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import os
import json
from typing import Any

from dotenv import load_dotenv

load_dotenv()

assert os.environ.get("GOOGLE_API_KEY"), "Please add a GOOGLE_API_KEY to the .env file."

from agentic_lci.llm_adapters.interface import LLM, T, Literal, encode_image

try:
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=os.environ["GOOGLE_API_KEY"])
except Exception as e:  # pragma: no cover - best-effort import
    raise Exception(
        "google.generativeai is not available. Install the package or configure the adapter."
    ) from e


class GeminiLLM(LLM[dict]):
    def __init__(self, model: str = "gemini-1.0"):
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

    def _flatten_messages_to_prompt(self, messages: list[dict]) -> str:
        parts: list[str] = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", [])
            for item in content:
                if item.get("type") == "input_text":
                    parts.append(f"[{role}] {item.get('text')}")
                elif item.get("type") == "input_image":
                    img = item.get("image_url")
                    parts.append(f"[{role}] <image: {img}>")
                else:
                    # best-effort stringify unknown content
                    parts.append(f"[{role}] {json.dumps(item)}")
        return "\n\n".join(parts)

    def unstructured_output(
        self, messages: list[dict], use_mcp: bool = False
    ) -> tuple[str, int]:
        """
        Returns (text, tokens). Tokens is best-effort (0 if not reported).
        """
        # TODO: add mcp support
        prompt = self._flatten_messages_to_prompt(messages)
        response = genai.generate_text(model=self.model, prompt=prompt)

        text = ""
        tokens = 0
        try:
            text = response.result or ""
        except Exception:
            try:
                text = str(response)
            except Exception:
                text = ""

        try:
            if hasattr(response, "usage") and getattr(response, "usage") is not None:
                tokens = int(getattr(response.usage, "total_tokens", 0))
        except Exception:
            tokens = 0

        messages.append(self.create_message("assistant", text))
        return text, tokens

    def structured_output(
        self, messages: list[dict], output_type: type[T]
    ) -> tuple[T, int]:
        """
        Requests the model to output JSON that matches the Pydantic model `output_type`
        and parses it locally. Returns (parsed_obj, tokens).
        """
        prompt = self._flatten_messages_to_prompt(messages)
        # ask model to output only JSON that matches the schema
        schema_hint = ""
        try:
            # if the output_type is a pydantic model, include its schema for better parsing
            schema_hint = output_type.schema_json(indent=None)  # type: ignore
        except Exception:
            schema_hint = ""

        full_prompt = (
            f"{prompt}\n\nRespond with a JSON object matching this schema:\n{schema_hint}\n\n"
            "Do not include any explanatory text, only the JSON."
        )

        response = genai.generate_text(model=self.model, prompt=full_prompt)  # type: ignore

        text = ""
        tokens = 0
        try:
            text = response.result or ""
        except Exception:
            text = str(response)

        try:
            if hasattr(response, "usage") and getattr(response, "usage") is not None:
                tokens = int(getattr(response.usage, "total_tokens", 0))
        except Exception:
            tokens = 0

        # attempt to parse JSON from the response text
        parsed_json: Any
        try:
            parsed_json = json.loads(text)
        except Exception:
            # fallback: try to extract a JSON substring
            start = text.find("{")
            end = text.rfind("}") + 1
            try:
                parsed_json = json.loads(text[start:end])
            except Exception as e:
                raise ValueError(
                    f"Failed to parse JSON from model output: {e}\nOutput was:\n{text}"
                )

        # parse into the requested pydantic type if possible
        try:
            parsed_obj: T = output_type.parse_obj(parsed_json)  # type: ignore
        except Exception:
            # if output_type is raw dict-like, just return parsed_json
            parsed_obj = parsed_json  # type: ignore

        messages.append(self.create_message("assistant", text))
        return parsed_obj, tokens


if __name__ == "__main__":
    # example/test usage
    from pydantic import BaseModel

    class SimpleStructure(BaseModel):
        footprint: int
        unit: str

    llm: LLM = GeminiLLM()
    messages = [
        llm.create_message(
            "user",
            "use the lookup_db_info tool on the iPhone 13 to estimate its carbon footprint",
        )
    ]
    print(llm.unstructured_output(messages, True))
    print(llm.structured_output(messages, SimpleStructure))
