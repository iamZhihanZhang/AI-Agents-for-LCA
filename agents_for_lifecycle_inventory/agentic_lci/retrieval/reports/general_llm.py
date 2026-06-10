# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - General report parser using LLM

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import datetime

from agentic_lci.llm_adapters import LLM, default_llm
from agentic_lci.retrieval.web.text import url_to_markdown


def parse_general(url, llm: LLM = default_llm):
    from agentic_lci.schemas.electronics_gwp import (
        PROMPT_DESCRIPTION,
        ProductReportModel,
    )

    document_txt = url_to_markdown(url)

    report, tokens = llm.structured_output(
        [
            {
                "role": "system",
                "content": "You are a professional environmental scientist and adept at reading Product Carbon Footprint reports.",
            },
            {
                "role": "user",
                "content": f"""
                    Reply with the important numbers and values from the reports you are given in format. Do not say anything else.

                    {PROMPT_DESCRIPTION}

                    Here is the report content:
                    {document_txt}
                """,
            },
        ],
        ProductReportModel,
    )
    # TODO: add back token usage logging
    report.add_date = datetime.datetime.now().strftime("%Y-%m-%d")
    report.add_method = "General Report Parser"
    report.source_urls = [url]
    return report
