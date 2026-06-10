# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - LCI schema for electronics components from the DeltaLCA paper (https://arxiv.org/abs/2311.09611)

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

from typing import Dict, Literal, Optional
from pydantic import BaseModel, Field, RootModel, ConfigDict


# --------------------
# Sub-models
# --------------------
class SizeModel(BaseModel):
    Width: float = Field(..., gt=0)
    Height: float = Field(..., gt=0)
    Unit: Literal["inches"]


class BoardModel(BaseModel):
    Size: SizeModel


class PassiveComponentCounts(RootModel[Dict[str, int]]):
    """Uses imperial size codes and includes counts for each. Example: { "0201": 12, "0402": 4 }"""

    pass


class ICEntry(BaseModel):
    Name: str
    Size: SizeModel
    Count: int = Field(..., ge=1)


class ICComponent(RootModel[Dict[str, ICEntry]]):
    """
    Example:
    "abc123": {
        "Name": "Unknown",
        "Size": { "Width": 0.2, "Height": 0.1, "Unit": "inches" },
        "Count": 1
    }
    """

    pass


class DetectedComponent(BaseModel):
    DetectionID: str
    Size: SizeModel
    Confidence: float = Field(..., ge=0, le=1)


# --------------------
# Main model
# --------------------
class DeltaLCAInput(BaseModel):
    board: BoardModel

    capacitor: Optional[PassiveComponentCounts] = Field(alias="Capacitor", default=None)
    electrolytic_capacitor: Optional[PassiveComponentCounts] = Field(
        alias="Electrolytic Capacitor", default=None
    )
    inductor: Optional[PassiveComponentCounts] = Field(alias="Inductor", default=None)
    resistor: Optional[PassiveComponentCounts] = Field(alias="Resistor", default=None)
    ic: Optional[ICComponent] = Field(alias="IC", default=None)

    model_config = ConfigDict(extra="allow")
    __pydantic_extra__: dict[str, list[DetectedComponent]] = Field(init=False)


# --------------------
# Parsing utilities
# --------------------
PROMPT_DESCRIPTION = """
Example:
{
    "board": {
        "Size": {
            "Width": 1.5090217506252226,
            "Height": 2.3222787705496994,
            "Unit": "inches",
        }
    },
    "IC": {
        "b63820a6-56d5-4671-acf0-570fc8710d9a": {
            "Name": "Unknown",
            "Size": {
                "Width": 0.456807653141718,
                "Height": 0.2610329446524103,
                "Unit": "inches",
            },
            "Count": 1,
        },
        "5facc85f-e57b-4066-b71d-d4a08faf2c91": {
            "Name": "Unknown",
            "Size": {
                "Width": 0.4894367712232693,
                "Height": 0.40460106421123593,
                "Unit": "inches",
            },
            "Count": 1,
        },
        "4cae64ce-7ef3-4cee-b87f-27d20caa5916": {
            "Name": "Unknown",
            "Size": {
                "Width": 0.16967141402406669,
                "Height": 0.2545071210361,
                "Unit": "inches",
            },
            "Count": 1,
        },
        "b0f98793-d035-4451-abdc-2ae3968e8c32": {
            "Name": "Unknown",
            "Size": {
                "Width": 0.4111268878275462,
                "Height": 0.3328170044318231,
                "Unit": "inches",
            },
            "Count": 1,
        },
        "6e708e6a-0a64-4b7b-9c84-b05ddc2bf81e": {
            "Name": "Unknown",
            "Size": {
                "Width": 0.31976535719920257,
                "Height": 0.41765271144385646,
                "Unit": "inches",
            },
            "Count": 1,
        },
        "bcfc6719-4293-4b05-ae72-8f960af7a2fc": {
            "Name": "Unknown",
            "Size": {
                "Width": 0.1370422959425154,
                "Height": 0.15661976679144618,
                "Unit": "inches",
            },
            "Count": 1,
        },
        "2e0f5bc8-b6da-45dc-b709-5b889903583a": {
            "Name": "Unknown",
            "Size": {
                "Width": 0.15661976679144618,
                "Height": 0.1500939431751359,
                "Unit": "inches",
            },
            "Count": 1,
        },
        "bd0b4c08-976f-48f3-ae13-ea068ff4c2fc": {
            "Name": "Unknown",
            "Size": {
                "Width": 0.10441317786096412,
                "Height": 0.11093900147727437,
                "Unit": "inches",
            },
            "Count": 1,
        },
        "f9a0bf00-9f1d-4162-b25a-8f727af79491": {
            "Name": "Unknown",
            "Size": {
                "Width": 0.29366206273396156,
                "Height": 0.3001878863502718,
                "Unit": "inches",
            },
            "Count": 1,
        },
        "3e31eda2-4973-412e-8be5-31c8a48b0790": {
            "Name": "Unknown",
            "Size": {
                "Width": 0.26755876826872055,
                "Height": 0.07830988339572309,
                "Unit": "inches",
            },
            "Count": 1,
        },
    },
    "Connector": [
        {
            "DetectionID": "edc9b57a-8097-4d37-bfb8-67bb2ef3d33e",
            "Size": {
                "Width": 0.18924888487299746,
                "Height": 0.2610329446524103,
                "Unit": "inches",
            },
            "Confidence": 0.9081336855888367,
        },
        {
            "DetectionID": "989c19e5-5ccc-44cd-bcbc-16facc194a1f",
            "Size": {
                "Width": 0.15661976679144618,
                "Height": 0.10441317786096412,
                "Unit": "inches",
            },
            "Confidence": 0.5597102642059326,
        },
    ],
    "Resistor Network": [
        {
            "DetectionID": "b6eb5457-2acc-4bc2-91a5-29af89d334c6",
            "Size": {
                "Width": 0.29366206273396156,
                "Height": 0.05873241254679231,
                "Unit": "inches",
            },
            "Confidence": 0.4618871212005615,
        }
    ],
    "Capacitor Jumper": [
        {
            "DetectionID": "bc1d891b-ef16-492c-84ad-d09f494a9e7f",
            "Size": {
                "Width": 0.14356811955882565,
                "Height": 0.10441317786096412,
                "Unit": "inches",
            },
            "Confidence": 0.34690913558006287,
        }
    ],
    "Electrolytic Capacitor": {"2010": 1},
}

Unavailable values/components are left out.
"""
