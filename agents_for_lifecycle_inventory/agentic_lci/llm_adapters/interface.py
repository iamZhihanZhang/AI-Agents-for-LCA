# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - LLM interface

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import base64
from typing import TypeVar, Generic, Literal

T = TypeVar("T")
MSG = TypeVar("MSG")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class LLM(Generic[MSG]):
    def create_message(
        self,
        role: Literal["system", "user", "assistant"],
        prompt: str | None = None,
        image_url: str | None = None,
    ) -> MSG:
        raise NotImplementedError()

    def unstructured_output(
        self, messages: list[MSG], use_mcp: bool = False
    ) -> tuple[str, int]:
        raise NotImplementedError()

    def structured_output(
        self, messages: list[MSG], output_type: type[T]
    ) -> tuple[T, int]:
        raise NotImplementedError()
