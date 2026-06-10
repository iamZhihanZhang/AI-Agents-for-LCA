# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - General purpose LCI schema from the Living Sustainability paper (https://dl.acm.org/doi/10.1145/3749488)

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


# --------------------
# Sub-models
# --------------------
class FunctionalUnit(BaseModel):
    """The reference quantity for all LCA result"""

    description: str = Field(
        description='Human - readable definition (e.g. "1 unit of product ")'
    )
    source: str = Field(description='Whether it was "provided" or "inferred"')


class Process(BaseModel):
    name: str
    inputs: Optional[List[str]] = []
    power: Optional[float] = None
    power_unit: Optional[str] = None
    power_original: Optional[float] = None
    power_original_unit: Optional[str] = None
    time: Optional[float] = None
    time_unit: Optional[str] = None
    time_original: Optional[float] = None
    time_original_unit: Optional[str] = None
    power_source: Optional[str] = None
    time_source: Optional[str] = None
    energy: Optional[float] = None
    energy_unit: Optional[str] = None
    energy_source: Optional[str] = None
    outputs: Optional[List[str]] = []
    waste: Optional[List[str]] = []
    location: Optional[str] = None
    carbon_emission_factor: Optional[str] = None
    carbon_emission_source: Optional[str] = None
    index: int


class MaterialRatio(BaseModel):
    name: str
    ratio_value: float
    unit: str
    ratio_value_source: str = Field(
        description='Whether it was "provided" or "inferred"'
    )
    unit_source: str = Field(description='Whether it was "provided" or "inferred"')
    carbon_emission_factor: str
    carbon_emission_source: str = Field(description="URL")
    index: int


class RelatedMaterial(BaseModel):
    """Materials defined via ratios or mixtures (e.g. 1:2 mix of A and B)"""

    text_source: str
    ratio: List[MaterialRatio]
    processes: Optional[List[Process]] = Field(
        default=[], description="Subprocesses applied to raw materials"
    )


class IndependentMaterial(BaseModel):
    """Materials directly specified by quantity"""

    name: str
    amount: float
    unit: str
    amount_source: str = Field(description='Whether it was "provided" or "inferred"')
    unit_source: str = Field(description='Whether it was "provided" or "inferred"')
    carbon_emission_factor: str
    carbon_emission_source: str = Field(description="URL")
    index: int


class Production(BaseModel):
    related_materials: Optional[List[RelatedMaterial]] = []
    independent_material: Optional[List[IndependentMaterial]] = []
    processes: Optional[List[Process]] = Field(
        default=[], description="General processing (e.g. assembly)"
    )


class Weight(BaseModel):
    value: Optional[float] = None
    unit: Optional[str] = None
    weight_source: Optional[str] = Field(
        description='Whether it was "provided" or "inferred"', default=None
    )


class TransportSegment(BaseModel):
    """Transport routes for moving materials or products"""

    from_location: Optional[str] = None
    to_location: Optional[str] = None
    weight: Optional[Weight] = None
    vehicle_type: Optional[str] = None
    vehicle_type_source: Optional[str] = Field(
        description='Whether it was "provided" or "inferred"', default=None
    )
    carbon_emission_factor: Optional[str] = None
    carbon_emission_source: Optional[str] = Field(description="URL", default=None)
    index: int


class Transport(BaseModel):
    segments: Optional[List[TransportSegment]] = []


class UseOperation(BaseModel):
    name: str
    power: Optional[float] = None
    power_unit: Optional[str] = None
    power_original: Optional[float] = None
    power_original_unit: Optional[str] = None
    time: Optional[float] = None
    time_unit: Optional[str] = None
    time_original: Optional[float] = None
    time_original_unit: Optional[str] = None
    power_source: Optional[str] = Field(
        description='Whether it was "provided" or "inferred"', default=None
    )
    time_source: Optional[str] = Field(
        description='Whether it was "provided" or "inferred"', default=None
    )
    energy: Optional[float] = None
    energy_unit: Optional[str] = None
    energy_source: Optional[str] = Field(
        description='Whether it was "provided" or "inferred"', default=None
    )
    location: Optional[str] = None
    carbon_emission_factor: Optional[str] = None
    carbon_emission_source: Optional[str] = Field(description="URL", default=None)
    index: int


class Use(BaseModel):
    """How the product consumes energy in operation or use"""

    operations: Optional[List[UseOperation]] = []


# --------------------
# Main model
# --------------------
class LCAInput(BaseModel):
    functional_unit: FunctionalUnit
    production: Production
    transport: Transport
    use: Use
    explanation: Optional[str] = Field(
        default=None,
        description="Plain - language summary for the general public in UI information panel",
    )


# --------------------
# Parsing utilities
# --------------------
PROMPT_DESCRIPTION = """
Format:
Your structured output must have a functional_unit, production, transport, use, and explanation.

The functional_unit defines the reference quantity for all LCA results.
- description: a human-readable definition (e.g. "1 unit of product", "1 smartphone used for 1 year")
- source: whether the functional unit was "provided" or "inferred"

The production section describes materials and processes used to manufacture the product:
    - related_materials are materials defined via ratios or mixtures (e.g. alloys, composites, or blends)
        - Each related material includes a text_source explaining the mixture
        - ratio defines the material composition
        - processes are optional subprocesses applied to raw materials
    - independent_material are materials directly specified by quantity
    - processes are general production or assembly processes applied at the product level

Each process may include:
    - inputs, outputs, and waste
    - power, time, and/or energy (with original values if conversions were made)
    - location and carbon emission factors
    - index must be unique within its list

The transport section describes transportation of materials or finished products:
    - segments represent individual transport legs
    - include from_location, to_location, vehicle_type, and transported weight when known
    - index must be unique within the transport segments

The use section describes how the product consumes energy during operation:
    - operations represent distinct usage modes or phases
    - power, time, and energy may be provided or inferred
    - include location when relevant
    - index must be unique within use operations

The explanation is a plain-language summary intended for a general audience UI panel.

Example 1:
{
  "functional_unit": {
    "description": "1 smartphone used for 3 years",
    "source": "inferred"
  },

  "production": {
    "related_materials": [
      {
        "text_source": "Aluminum alloy enclosure",
        "ratio": [
          {
            "name": "Aluminum",
            "ratio_value": 0.95,
            "unit": "mass fraction",
            "ratio_value_source": "inferred",
            "unit_source": "inferred",
            "carbon_emission_factor": "8.24 kg CO2e/kg",
            "carbon_emission_source": "https://ecoinvent.org",
            "index": 0
          },
          {
            "name": "Magnesium",
            "ratio_value": 0.05,
            "unit": "mass fraction",
            "ratio_value_source": "inferred",
            "unit_source": "inferred",
            "carbon_emission_factor": "26 kg CO2e/kg",
            "carbon_emission_source": "https://ecoinvent.org",
            "index": 1
          }
        ],
        "processes": [
          {
            "name": "Die casting",
            "inputs": ["Aluminum alloy"],
            "power": 15,
            "power_unit": "kW",
            "time": 0.2,
            "time_unit": "h",
            "energy": 3,
            "energy_unit": "kWh",
            "energy_source": "inferred",
            "location": "China",
            "carbon_emission_factor": "0.58 kg CO2e/kWh",
            "carbon_emission_source": "https://www.iea.org",
            "index": 0
          }
        ]
      }
    ],

    "independent_material": [
      {
        "name": "Lithium-ion battery",
        "amount": 0.045,
        "unit": "kg",
        "amount_source": "inferred",
        "unit_source": "provided",
        "carbon_emission_factor": "95 kg CO2e/kg",
        "carbon_emission_source": "https://www.nature.com",
        "index": 0
      }
    ],

    "processes": [
      {
        "name": "Final assembly",
        "inputs": ["Electronic components", "Mechanical parts"],
        "time": 0.5,
        "time_unit": "h",
        "location": "China",
        "carbon_emission_factor": "0.58 kg CO2e/kWh",
        "carbon_emission_source": "https://www.iea.org",
        "index": 1
      }
    ]
  },

  "transport": {
    "segments": [
      {
        "from_location": "China",
        "to_location": "Europe",
        "weight": {
          "value": 0.21,
          "unit": "kg",
          "weight_source": "inferred"
        },
        "vehicle_type": "container ship",
        "vehicle_type_source": "inferred",
        "carbon_emission_factor": "0.01 kg CO2e/ton-km",
        "carbon_emission_source": "https://ecoinvent.org",
        "index": 0
      }
    ]
  },

  "use": {
    "operations": [
      {
        "name": "Daily smartphone usage",
        "power": 5,
        "power_unit": "W",
        "time": 3,
        "time_unit": "h/day",
        "energy": 0.005,
        "energy_unit": "kWh/day",
        "energy_source": "inferred",
        "location": "Europe",
        "carbon_emission_factor": "0.25 kg CO2e/kWh",
        "carbon_emission_source": "https://www.iea.org",
        "index": 0
      }
    ]
  },

  "explanation": "This assessment estimates the environmental impact of producing, transporting, and using a smartphone over a typical three-year lifespan."
}

Example 2:
{
  "functional_unit": {
    "description": "1 electronic device",
    "source": "provided"
  },

  "production": {
    "related_materials": [],
    "independent_material": [],
    "processes": []
  },

  "transport": {
    "segments": []
  },

  "use": {
    "operations": []
  },

  "explanation": null
}

Reasoning Rules:
- Use SI units where possible (kg, kWh, hours)
- Preserve original values and units if conversions are made
- Emission factors must include a source URL when known
- index values must be unique within each list
- Lists may be empty ([]) if data is unknown
- Null/None are used for unknown values
- Do not invent precise numbers unless reasonably inferred
"""
