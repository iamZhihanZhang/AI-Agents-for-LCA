# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - The schema defining and critiquing agent

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

from pydantic import BaseModel, Field

from agentic_lci.schemas import SCHEMAS, SCHEMA_SUMMARIES
from agentic_lci.llm_adapters import LLM, default_llm


class Schema(BaseModel):
    index: int = Field(..., ge=0, le=len(SCHEMAS) - 1)
    product_name: str
    instructions_for_stakeholder: str


class Critique(BaseModel):
    is_complete: bool
    feedback: str


class LCAAgent:
    def __init__(self, llm: LLM = default_llm):
        self.llm = llm

    def design_schema(
        self, query: str | None = None, image: str | None = None
    ) -> Schema:
        assert query or image, "At least one of text and image prompts must be provided"
        # TODO: allow generating schema on the fly
        messages = [
            self.llm.create_message(
                "system",
                f"""
                You are a detail-oriented Life Cycle Assessment (LCA) specialist whose goal is to select one of the below {len(SCHEMAS)} schemas
                for a Life Cycle Inventory (LCI) to best describe the product requested by the user. You will subsequently work with the 
                Stakeholders to fill out the schema so please write a standalone description of the requested product and what to do to the
                Stakeholders in addition to choosing a schema. Here are the schemas:
                
                {SCHEMA_SUMMARIES}
                """,
            ),
            self.llm.create_message("user", query, image),
        ]
        # TODO: log token usage
        return self.llm.structured_output(messages, Schema)[0]

    def critique(self, inventory: BaseModel, schema: Schema) -> Critique:
        messages = [
            self.llm.create_message(
                "system",
                f"""
                You are a detail-oriented Life Cycle Assessment (LCA) specialist who is reviewing the Life Cycle Inventory (LCI)
                from the Stakeholder for the {schema.product_name} product. Please review it for completion, coherency, and accuracy. 
                Only mark it as complete if the Stakeholder's version is done and does not need further iteration. Do not yourself
                attempt to modify the LCI. Null values are not complete unless they really cannot be determined.
                
                The LCI must follow this schema:
                {SCHEMAS[schema.index]['description']}
                """,
            ),
            self.llm.create_message(
                "user",
                f"As stakeholder, I have filled out the inventory this much so far:\n{inventory.model_dump_json()}",
            ),
        ]
        # TODO: log token usage
        return self.llm.structured_output(messages, Critique)[0]
