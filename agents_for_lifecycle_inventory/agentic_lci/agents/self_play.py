# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - The main self-play loop between the agents

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

from .lca_agent import LCAAgent
from .stakeholder_agent import StakeholderAgent
from agentic_lci.schemas import SCHEMAS
from agentic_lci.llm_adapters import LLM, default_llm


def run_self_play(
    query: str | None = None,
    image: str | None = None,
    llm: LLM = default_llm,
    max_rounds=5,
):
    lca = LCAAgent(llm)
    stakeholder = StakeholderAgent(llm)

    schema = lca.design_schema(query, image)
    print(f"LCA Agent chose schema: {SCHEMAS[schema.index]['model'].__name__}")
    inventory = None
    critique = None

    for round in range(max_rounds):
        print(f"Round {round + 1}...")
        inventory = stakeholder.fill(schema, inventory, critique)

        critique = lca.critique(inventory, schema)
        if critique.is_complete:
            break

    return inventory, critique
