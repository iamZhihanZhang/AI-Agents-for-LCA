# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - LCI schema for electronics products

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

from typing import Optional, Any
from pydantic import BaseModel, Field


# --------------------
# Sub-models
# --------------------
class StorageSpec(BaseModel):
    type: Optional[str] = Field(..., description="HDD | SSD | eMMC | ...")
    capacity_gb: Optional[float]


class MemorySpec(BaseModel):
    type: Optional[str] = Field(
        ..., description="DDR3 | DDR4 | DDR5 | LPDDR4 | GDDR6 | ..."
    )
    capacity_gb: Optional[float]


class BatterySpec(BaseModel):
    capacity: Optional[float]
    unit: Optional[str] = Field(..., description="mAh | Wh")


class Resolution(BaseModel):
    width_px: Optional[int]
    height_px: Optional[int]


class DisplaySpec(BaseModel):
    diagonal_size_in: Optional[float]
    width_mm: Optional[float]
    height_mm: Optional[float]
    panel_type: Optional[str] = Field(..., description="OLED | IPS | VA | TN | ...")
    resolution: Resolution
    max_brightness_nits: Optional[float]
    refresh_rate_hz: Optional[float]


class DimensionSpec(BaseModel):
    width_mm: Optional[float]
    height_mm: Optional[float]
    depth_mm: Optional[float]


class GeneralSpecs(BaseModel):
    product_class: Optional[str] = Field(
        ..., description="Smartphone | Laptop | Desktop | Display | ..."
    )
    product_name: Optional[str]
    assembly_location: Optional[str]

    storage: StorageSpec
    memory: MemorySpec
    battery: BatterySpec
    display: DisplaySpec

    dimension: DimensionSpec
    weight_g: Optional[float]


class PCB(BaseModel):
    name: Optional[str]
    length_mm: Optional[float]
    width_mm: Optional[float]
    area_mm2: Optional[float]
    layers: Optional[int]
    material: Optional[str] = Field(..., description="FR4 | Polyimide | ...")
    emission_factor: Optional[float] = Field(
        ..., description="Number of g CO2e per component"
    )
    emission_source: Optional[str] = Field(
        ..., description="The url source(s) for the emmision factor"
    )


class Processor(BaseModel):
    name: Optional[str]
    type: Optional[str] = Field(..., description="CPU | GPU | SoC | MCU | ...")
    technology_node_nm: Optional[float]
    package_type: Optional[str] = Field(..., description="LGA | BGA | QFN | ...")
    length_mm: Optional[float]
    width_mm: Optional[float]
    package_area_mm2: Optional[float]
    die_area_mm2: Optional[float]
    emission_factor: Optional[float] = Field(
        ..., description="Number of g CO2e per component"
    )
    emission_source: Optional[str] = Field(
        ..., description="The url source(s) for the emmision factor"
    )


class KeyValueSpec(BaseModel):
    key: str
    value: str | float | int | bool


class DiscreteComponent(BaseModel):
    name: Optional[str]
    component_type: Optional[str] = Field(
        ...,
        description="resistor | capacitor | inductor | transistor | OpAmp | connector | ...",
    )
    count: Optional[int]
    package_type: Optional[str]
    length_mm: Optional[float]
    width_mm: Optional[float]
    package_area_mm2: Optional[float]

    # Flexible per-component metadata
    additional_specs: Optional[list[KeyValueSpec]] = Field(
        ...,
        description="values like the capacitance_uf, sensor_type, connector_type, a general description, etc.",
    )

    emission_factor: Optional[float] = Field(
        ..., description="Number of g CO2e per component"
    )
    emission_source: Optional[str] = Field(
        ..., description="The url source(s) for the emmision factor"
    )


class ElectronicComponents(BaseModel):
    processors: list[Processor] = []
    discrete_components: list[DiscreteComponent] = []


class MechanicalDimensions(BaseModel):
    width_mm: Optional[float]
    height_mm: Optional[float]
    depth_mm: Optional[float]
    thickness_mm: Optional[float]


class MechanicalComponent(BaseModel):
    name: Optional[str]
    component_type: Optional[str] = Field(
        ...,
        description="enclosure | fan | structural | ...",
    )
    count: Optional[int]
    material: Optional[str] = Field(..., description="Aluminum | Plastic | ...")
    manufacturing_process: Optional[str] = Field(
        ...,
        description="CNC Machining | Die Casting | Injection Molding | Stamping | ...",
    )
    weight_g: Optional[float]
    dimensions: MechanicalDimensions
    emission_factor: Optional[float] = Field(
        ..., description="Number of g CO2e per component"
    )
    emission_source: Optional[str] = Field(
        ..., description="The url source(s) for the emmision factor"
    )


class Production(BaseModel):
    pcb: list[PCB] = []
    electronic_components: ElectronicComponents
    mechanical_components: list[MechanicalComponent] = []


# --------------------
# Main model
# --------------------
class ElectronicProduct(BaseModel):
    functional_unit: Optional[str] = Field(
        ...,
        description="the reference quantity for all LCI values, e.g., '1 unit of electronic product' or '1 phone'",
    )
    general_specs: GeneralSpecs
    production: Production
    comments: str


# --------------------
# Parsing utilities
# --------------------
PROMPT_DESCRIPTION = """
Format:
Your structured output must have a functional_unit, general_specs, and production components.

The functional_unit is human-readable reference quantity for all LCI values. E.g., "1 unit of electronic product", "1 smartphone", "1 laptop computer".

The general_specs describe what the product is, its form factor, and high-level hardware specs.

The production describe physical components contributing to manufacturing emissions:
    - Each PCB entry represents one board.
    - Electronic Components are split into
        - processors (Large ICs such as CPUs, GPUs, SoCs, MCUs)
        - discrete_components (Smaller components such as: Resistors, Capacitor, Inductor, Sensor, Connector, OpAmps)
        - Use additional_specs for component-specific attributes.
    - Mechanical Components are non-electronic physical parts such as: Enclosures, Frames, Fans, Structural brackets

Example 1: {
  "functional_unit": "1 smartphone",

  "general_specs": {
    "product_class": "Smartphone",
    "product_name": "Example Phone X",
    "assembly_location": "China",

    "storage": {
      "type": "UFS",
      "capacity_gb": 256
    },

    "memory": {
      "type": "LPDDR5",
      "capacity_gb": 12
    },

    "battery": {
      "capacity": 4800,
      "unit": "mAh"
    },

    "display": {
      "diagonal_size_in": 6.7,
      "width_mm": 68.5,
      "height_mm": 152.3,
      "panel_type": "OLED",
      "resolution": {
        "width_px": 1440,
        "height_px": 3200
      },
      "max_brightness_nits": 1200,
      "refresh_rate_hz": 120
    },

    "dimension": {
      "width_mm": 75.8,
      "height_mm": 162.9,
      "depth_mm": 8.9
    },

    "weight_g": 210
  },

  "production": {
    "pcb": [
      {
        "name": "Main logic board",
        "length_mm": 90,
        "width_mm": 45,
        "area_mm2": 4050,
        "layers": 10,
        "material": "FR4",
        "emission_factor": 120,
        "emission_source": "https://ecoinvent.org"
      }
    ],

    "electronic_components": {
      "processors": [
        {
          "name": "Snapdragon 8 Gen 2",
          "type": "SoC",
          "technology_node_nm": 4,
          "package_type": "BGA",
          "length_mm": 15,
          "width_mm": 15,
          "package_area_mm2": 225,
          "die_area_mm2": 90,
          "emission_factor": 45,
          "emission_source": "https://www.tsmc.com"
        }
      ],

      "discrete_components": [
        {
          "name": "Ceramic capacitors",
          "component_type": "capacitor",
          "count": 450,
          "package_type": "0402",
          "length_mm": 1.0,
          "width_mm": 0.5,
          "package_area_mm2": 0.5,
          "additional_specs": [
            {"capacitance_uf": 0.1}
          ],
          "emission_factor": 0.02,
          "emission_source": "https://ecoinvent.org"
        }
      ]
    },

    "mechanical_components": [
      {
        "name": "Aluminum enclosure",
        "component_type": "enclosure",
        "count": 1,
        "material": "Aluminum",
        "manufacturing_process": "CNC Machining",
        "weight_g": 35,
        "dimensions": {
          "width_mm": 75.8,
          "height_mm": 162.9,
          "depth_mm": 8.9,
          "thickness_mm": 0.8
        },
        "emission_factor": 180,
        "emission_source": "https://www.aluminium.org"
      }
    ]
  }
}


Example 2: {
    "functional_unit": "1 electronic display module",
    "general_specs": {
        "product_class": "Display",
        "product_name": None,
        "assembly_location": None,
        "storage": {"type": None, "capacity_gb": None},
        "memory": {"type": None, "capacity_gb": None},
        "battery": {"capacity": None, "unit": None},
        "display": {
            "diagonal_size_in": 15.6,
            "width_mm": None,
            "height_mm": None,
            "panel_type": "IPS",
            "resolution": {"width_px": 1920, "height_px": 1080},
            "max_brightness_nits": None,
            "refresh_rate_hz": 60,
        },
        "dimension": {"width_mm": None, "height_mm": None, "depth_mm": None},
        "weight_g": None,
    },
    "production": {
        "pcb": [],
        "electronic_components": {"processors": [], "discrete_components": []},
        "mechanical_components": [],
    },
}

Reasoning Rules:
- Use millimeters (mm) for dimensions
- Use grams (g) for weights
- Emission factors are per component, not totals
- URLs should be real when known, otherwise left out
- Lists may be empty ([]) if no components exist or is unknown
- Null/None are used for unknown values.
"""
