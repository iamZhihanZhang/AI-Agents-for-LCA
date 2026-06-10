# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Report Parser

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

from .vendor import *
from .general_llm import parse_general
from agentic_lci.schemas.electronics_gwp import ProductReportModel


def parse_report(url: str, include_general=True) -> ProductReportModel | None:
    """Cycle through the report parsers until one works."""

    res = None
    parsers = [
        parse_apple,
        parse_asus,
        parse_dell,
        parse_google,
        parse_hp,
        parse_hpe,
        parse_huawei,
        parse_lenovo,
        parse_microsoft,
    ]
    if include_general:
        parsers.append(parse_general)
    for parser in parsers:
        try:
            # TODO: optimize by only fetching URL content once at the beginning
            res = parser(url)
            if res:
                break
        except Exception as e:
            continue
    return res
