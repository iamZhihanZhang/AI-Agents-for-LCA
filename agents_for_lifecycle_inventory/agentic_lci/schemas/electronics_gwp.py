# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - LCI schema for electronics carbon footprint

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import re
from typing import List, Optional, Literal, TYPE_CHECKING
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from agentic_lci.retrieval.reports.vendor.boavizta.utils.data import (
        DeviceCarbonFootprint,
    )


# --------------------
# Sub-models
# --------------------
class ChipModel(BaseModel):
    CPU: Optional[str]
    GPU: Optional[str]
    SoC: Optional[str]


class CoreCountModel(BaseModel):
    CPU: Optional[int] = Field(None, ge=0)
    GPU: Optional[int] = Field(None, ge=0)
    SoC: Optional[int] = Field(None, ge=0)


class HardDriveModel(BaseModel):
    type: Optional[Literal["HDD", "SSD", "eMMC", "UFS", "NVMe"]]
    size_gb: Optional[float] = Field(..., ge=0)


class DisplayModel(BaseModel):
    size_in: Optional[float] = Field(..., ge=0)
    type: Optional[
        Literal[
            "OLED",
            "LCD",
            "miniLED",
            "microLED",
            "AMOLED",
            "POLED",
            "IPS",
            "VA",
            "TN",
        ]
    ]
    max_brightness_nits: Optional[float] = Field(..., ge=0)
    refresh_rate_hz: Optional[float] = Field(..., ge=0)


class DimensionModel(BaseModel):
    width_cm: Optional[float] = Field(..., ge=0)
    height_cm: Optional[float] = Field(..., ge=0)
    depth_cm: Optional[float] = Field(..., ge=0)


class GWPTopComponentsModel(BaseModel):
    pcb: Optional[float]
    system_component: Optional[float]
    display: Optional[float]
    battery: Optional[float]
    keyboard: Optional[float]
    hdd: Optional[float]
    ssd: Optional[float]
    storage: Optional[float]
    inductor: Optional[float]
    capacitor: Optional[float]
    ic: Optional[float]


class GWPModel(BaseModel):
    total_kg_CO2: Optional[float] = Field(..., ge=0)

    manufacturing_phase: Optional[float] = Field(..., ge=0, le=1)
    transport_phase: Optional[float] = Field(..., ge=0, le=1)
    use_phase: Optional[float] = Field(..., ge=0, le=1)
    end_of_life_phase: Optional[float] = Field(..., ge=0, le=1)

    electronics: Optional[float] = Field(..., ge=0, le=1)
    error: Optional[float] = Field(..., ge=0, le=1)

    top_components: Optional[GWPTopComponentsModel]
    other_components: Optional[float] = Field(..., ge=0, le=1)


# --------------------
# Main model
# --------------------
class ProductReportModel(BaseModel):
    product_class: Optional[
        Literal[
            "Smartphone",
            "Laptop",
            "Desktop",
            "Display",
            "Motherboard",
            "Graphics Card",
            "Smartwatch",
            "Console",
            "Router",
            "Mouse",
            "Keyboard",
            "Other",
        ]
    ]

    product_name: Optional[str]
    manufacturer: Optional[str]

    report_date: Optional[str]
    add_date: Optional[str]

    use_location: Optional[str]
    assembly_location: Optional[str]

    chip: Optional[ChipModel]
    number_of_cores: Optional[CoreCountModel]

    hard_drive: Optional[HardDriveModel]

    memory_gb: Optional[float] = Field(None, ge=0)
    battery_capacity_mah: Optional[float] = Field(None, ge=0)

    display: Optional[DisplayModel]

    yearly_household_energy_consumption_kWh: Optional[float] = Field(None, ge=0)
    lifetime_years: Optional[float] = Field(None, ge=0)

    dimension: Optional[DimensionModel]
    weight_kg: Optional[float] = Field(..., ge=0)

    GWP: Optional[GWPModel]

    comment: Optional[str]
    source_urls: Optional[List[str]]

    add_method: Optional[str]


# --------------------
# Parsing utilities
# --------------------
def parse_hard_drive(hd_raw: Optional[str]) -> Optional[HardDriveModel]:
    if not hd_raw:
        return None

    # detect type
    hd_type = None
    for typ in ["HDD", "SSD", "eMMC", "UFS", "NVMe"]:
        if typ.lower() in hd_raw.lower():
            hd_type = typ
            break

    # detect size
    size_gb = None
    if m := re.search(r"([0-9]+)[kK][bB]", hd_raw):
        size_gb = int(m.group(1)) / 1024 / 1024
    elif m := re.search(r"([0-9]+)[mM][bB]", hd_raw):
        size_gb = int(m.group(1)) / 1024
    elif m := re.search(r"([0-9]+)[gG][bB]", hd_raw):
        size_gb = int(m.group(1))
    elif m := re.search(r"([0-9]+)[tT][bB]", hd_raw):
        size_gb = int(m.group(1)) * 1024

    if hd_type is None or size_gb is None:
        return None

    return HardDriveModel(
        type=hd_type,  # type: ignore
        size_gb=size_gb,
    )


def from_report(report: "DeviceCarbonFootprint", url: str) -> ProductReportModel:
    data = report.data

    electronics = ProductReportModel(
        product_class=data.get("subcategory"),  # type: ignore
        product_name=data.get("name"),
        manufacturer=data.get("manufacturer"),
        report_date=data.get("report_date"),
        add_date=data.get("added_date"),
        use_location=data.get("use_location"),
        assembly_location=data.get("assembly_location"),
        chip=ChipModel(
            CPU=None,
            GPU=None,
            SoC=None,
        ),
        number_of_cores=CoreCountModel(
            CPU=data.get("number_cpu"),
            GPU=None,
            SoC=None,
        ),
        hard_drive=parse_hard_drive(data.get("hard_drive")),
        memory_gb=data.get("memory"),
        battery_capacity_mah=None,
        display=(
            DisplayModel(
                size_in=data.get("screen_size"),
                type=None,
                max_brightness_nits=None,
                refresh_rate_hz=None,
            )
            if data.get("screen_size") is not None
            else None
        ),
        yearly_household_energy_consumption_kWh=data.get("yearly_tec"),
        lifetime_years=data.get("lifetime"),
        dimension=DimensionModel(
            width_cm=None,
            height_cm=data.get("height"),
            depth_cm=None,
        ),
        weight_kg=data.get("weight"),
        GWP=GWPModel(
            total_kg_CO2=data.get("gwp_total"),
            manufacturing_phase=data.get("gwp_manufacturing_ratio"),
            transport_phase=data.get("gwp_transport_ratio"),
            use_phase=data.get("gwp_use_ratio"),
            end_of_life_phase=data.get("gwp_eol_ratio"),
            electronics=data.get("gwp_electronics_ratio"),
            error=data.get("gwp_error_ratio"),
            top_components=GWPTopComponentsModel(
                pcb=data.get("pcb_ratio"),
                system_component=data.get("system_component_ratio"),
                display=data.get("lcd_ratio"),
                battery=data.get("gwp_battery_ratio"),
                keyboard=data.get("keyboard_ratio"),
                hdd=data.get("gwp_hdd_ratio"),
                ssd=data.get("gwp_ssd_ratio"),
                storage=data.get("storage_ratio"),
                inductor=data.get("inductor_ratio"),
                capacitor=data.get("capacitor_ratio"),
                ic=data.get("ic_ratio"),
            ),
            other_components=data.get("gwp_othercomponents_ratio"),
        ),
        comment=data.get("comment"),
        source_urls=[url],
        add_method=data.get("add_method"),
    )

    return electronics


PROMPT_DESCRIPTION = """
Format:
{
    "product_class": string["Smartphone" | "Laptop" | "Desktop" | "Display" | "Motherboard" | "Graphics Card" | "Smartwatch", "Console" | "Router" | "Mouse" | "Keyboard" | "Other"],
    "product_name": string,
    "manufacturer": string,
    "report_date": string,
    "use_location": string,
    "assembly_location": string,
    "chip": {
        "CPU": "AMD 7945HX",
        "GPU": "Nvidia RTX 4070",
        "SoC": "Apple A14",
    },
    "number_of_cores": {
        "CPU": number of cores,
        "GPU": number of cores,
        "SoC": number of cores,
    },
    "hard_drive": {
        "type": string["HDD" | "SSD" | "eMMC" | "UFS" | "NVMe"],
        "size_gb": number of GB,
    },
    "memory_gb": number of GB,
    "battery_capacity_mah": number of mAh,
    "display": {
        "size_in": number of inches across diagonal,
        "type": string["OLED" | "LCD" | "miniLED" | "microLED" | "AMOLED" | "POLED" | "IPS" | "VA" | "TN"],
        "max_brightness_nits": number of nits,
        "refresh_rate_hz": number of Hz,
    },
    "yearly_household_energy_consumption_kWh": number of kWh,
    "lifetime_years": number of years,
    "dimension": {
        "width_cm": number of cm,
        "height_cm": number of cm,
        "depth_cm": number of cm,
    },
    "weight_kg": number of kg,
    "GWP": {
            "total_kg_CO2": number of kg of CO2e,
            "manufacturing_phase": ratio of total kg CO2e,
            "transport_phase": ratio of total kg CO2e,
            "use_phase": ratio of total kg CO2e,
            "end_of_life_phase": ratio of total kg CO2e,
            "electronics": ratio of total kg CO2e,
            "error": ratio of total kg CO2e,
            "top_components": {
                "pcb": ratio of total kg CO2e,
                "system_component": ratio of total kg CO2e,
                "display": ratio of total kg CO2e,
                "battery": ratio of total kg CO2e,
                "keyboard": ratio of total kg CO2e,
                "hdd": ratio of total kg CO2e,
                "ssd": ratio of total kg CO2e,
                "storage": ratio of total kg CO2e,
                "inductor": ratio of total kg CO2e,
                "capacitor": ratio of total kg CO2e,
                "ic": ratio of total kg CO2e,
            },
            "other_components": ratio of total kg CO2e,
    },
    "comment": string,
    "source_urls": string[],
    "add_date": string,
    "add_method": string,
}

Example:
{"product_class": "Smartphone", "product_name": "iPhone 12 Pro with 128GB", "manufacturer": "Apple", "report_date": "October 13, 2020", "use_location": "WW", "assembly_location": null, "chip": {"CPU": null, "GPU": null, "SoC": null}, "number_of_cores": {"CPU": null, "GPU": null, "SoC": null}, "hard_drive": {"name": "128GB SSD", "type": "SSD", "size_gb": 128}, "memory_gb": null, "battery_capacity_mah": null, "display": {"size": null, "type": null, "max_brightness_nits": null, "refresh_rate_hz": null}, "yearly_household_energy_consumption_kWh": null, "lifetime_years": 3.5, "dimension": {"width_cm": null, "height_cm": null, "depth_cm": null}, "weight_kg": null, "GWP": {"total_kg_CO2": 82.0, "manufacturing_phase": 0.86, "transport_phase": 0.02, "use_phase": 0.11, "end_of_life_phase": 0.01, "electronics": null, "error": null, "top_components": {"pcb": null, "system_component": null, "display": null, "battery": null, "keyboard": null, "hdd": null, "ssd": null, "storage": null, "inductor": null, "capacitor": null, "ic": null}, "other_components": null}, "comment": "iPhone 12 Pro 128GB (82kgCO2eq) - iPhone 12 Pro 256GB (93kgCO2eq) - iPhone 12 Pro 512GB (107kgCO2eq) - ", "source_urls": [], "add_date": "2024-06-28", "add_method": "PDF PCF Report Parser"}            

Another example:
{"product_class": "Laptop", "product_name": "G713PI", "manufacturer": "ASUS", "report_date": "April. 2024", "use_location": "Worldwide", "assembly_location": "China", "chip": {"CPU": null, "GPU": null, "SoC": null}, "number_of_cores": {"CPU": null, "GPU": null, "SoC": null}, "hard_drive": {"name": null, "type": null, "size_gb": null}, "memory_gb": null, "battery_capacity_mah": null, "display": {"size": 17.3, "type": null, "max_brightness_nits": null, "refresh_rate_hz": null}, "yearly_household_energy_consumption_kWh": 50.02, "lifetime_years": 4, "dimension": {"width_cm": null, "height_cm": null, "depth_cm": null}, "weight_kg": 2.8, "GWP": {"total_kg_CO2": 438.0, "manufacturing_phase": 0.65, "transport_phase": 0.055999999999999994, "use_phase": 0.293, "end_of_life_phase": 0.001, "electronics": null, "error": null, "top_components": {"pcb": 0.22, "system_component": 0.12, "display": null, "battery": null, "keyboard": 0.06, "hdd": null, "ssd": null, "storage": 0.06, "inductor": 0.08, "capacitor": null, "ic": 0.04}, "other_components": null}, "comment": null, "source_urls": [], "add_date": "2024-06-28", "add_method": "PDF PCF Report Parser"}

A third example:
{"product_class": "Laptop", "product_name": "Wyse 3030 Thin Client", "manufacturer": "Dell", "report_date": "August, 2018", "use_location": "EU", "assembly_location": "China", "chip": {"CPU": null, "GPU": null, "SoC": null}, "number_of_cores": {"CPU": null, "GPU": null, "SoC": null}, "hard_drive": {"name": null, "type": null, "size_gb": null}, "memory_gb": null, "battery_capacity_mah": null, "display": {"size": 0, "type": null, "max_brightness_nits": null, "refresh_rate_hz": null}, "yearly_household_energy_consumption_kWh": 19.2, "lifetime_years": 2.0, "dimension": {"width_cm": null, "height_cm": null, "depth_cm": null}, "weight_kg": 0.64, "GWP": {"total_kg_CO2": 70.0, "manufacturing_phase": 0.59, "transport_phase": 0.01, "use_phase": 0.39, "end_of_life_phase": 0.01, "electronics": 0.293, "error": 0.443, "top_components": {"pcb": null, "system_component": null, "display": null, "battery": null, "keyboard": null, "hdd": null, "ssd": null, "storage": null, "inductor": null, "capacitor": null, "ic": null}, "other_components": 0.147}, "comment": null, "source_urls": [], "add_date": "2024-06-28", "add_method": "PDF PCF Report Parser"}

Values that are not known are left as null/None.
"""
