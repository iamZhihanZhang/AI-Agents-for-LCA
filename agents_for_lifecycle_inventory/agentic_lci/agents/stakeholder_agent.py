# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - The information retrieval agent

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

from pydantic import BaseModel

from .lca_agent import Schema, Critique, SCHEMAS
from agentic_lci.llm_adapters import LLM, default_llm


class StakeholderAgent:
    def __init__(self, llm: LLM = default_llm):
        self.llm = llm

    def fill(
        self,
        schema: Schema,
        inventory: BaseModel | None,
        critique: Critique | None,
        turns=3,
    ) -> BaseModel:
        messages = [
            self.llm.create_message(
                "system",
                f"""
                You are a specialized stakeholder in the {schema.product_name} helping the Life Cycle Assessment (LCA) expert
                construct the Life Cycle Inventory (LCI) they have specified.

                Here are their overall instructions:
                {schema.instructions_for_stakeholder}
                
                Here is the LCI format specification:
                {SCHEMAS[schema.index]['description']}

                Here is what you currently have:
                {inventory.model_dump_json() if inventory else '{{}}'}

                You'll have {turns} turns to use the tool calls to fill out as many 
                of the missing values as possible and correct the newest critique given as a user message.
                """,
            ),
            (
                self.llm.create_message(
                    "user", "Critique from LCA expert: " + str(critique.feedback)
                )
                if critique
                else self.llm.create_message(
                    "user", "This is the first LCI attempt, no critique yet"
                )
            ),
        ]
        # TODO: log token usage
        for _ in range(turns):
            self.llm.unstructured_output(messages, use_mcp=True)
        return self.llm.structured_output(messages, SCHEMAS[schema.index]["model"])[0]
