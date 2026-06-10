# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Asus Report Parser

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import re
import datetime

from .boavizta.utils import data
from .boavizta.utils import pdf
from .boavizta.utils import text

import pytesseract
import pymupdf

_ASUS_LCA_PATTERNS = (
    # "Product Carbon Footprint Report" -> empty line -> "ASUS" -> product name and code -> empty line
    re.compile(r"Product Carbon Footprint Report\s*\n\s*ASUS\s*(?P<name>.*)\n"),
    # Report produced April. 2024
    re.compile(r"Report produced\s*(?P<date>[A-Z][a-z]*\s*\.?\s*[0-9]{4})"),
    # number with potential decimal -> space -> "kg" -> random text -> "Product weight"
    re.compile(r"(?P<weight>[0-9]*\.[0-9]*)\s*kg"),
    # number -> "years" -> random text -> "Lifetime"
    re.compile(r"(?P<lifetime>[0-9]*)\s*years"),
    # number with potential decimal -> "”" -> random text -> "Screen Size"
    re.compile(r"(?P<screen_size>[0-9]*\.[0-9]*)”"),
    # use location -> "Screen Size" -> "Use location"
    re.compile(r"(?P<use_location>[A-Za-z]*)\s*.*Screen Size.*Use location"),
    # location -> random text -> "Final Manufacturing Location"
    re.compile(
        r"(?P<assembly_location>[A-Za-z]*)\s*\n*\s*Final Manufacturing location"
    ),
    # "WHAT WE PRESENT" -> random text -> number of Kg CO2e -> xgco,e | kgco2e | kgco,e or any case combination -> newline
    re.compile(r"(?P<footprint>[0-9]*)\s*(?:xgco,e|kgco2e|kgco,e)"),
    # number with optional comma separator -> "numbers of" -> optional newline -> "smartphones charged"
    re.compile(
        r"(?P<smartphones_charged>[0-9]*,?[0-9]*)\s*numbers of.*smartphones charged"
    ),
    # number with optional decimal point -> "U.S. gallons of" -> newline -> "gasoline consumed"
    re.compile(
        r"(?P<gasoline_consumed>[0-9]*\.?[0-9]*)\s*U\.S\. gallons of.*gasoline consumed"
    ),
    # percentage with optional decimal point -> newline -> "Use"
    re.compile(r"(?P<gwp_use_ratio>[0-9]*\.?[0-9]*)%\s*Use"),
    # percentage with optional decimal point -> newline -> "Packaging & Ship"
    re.compile(r"(?P<gwp_transport_ratio>[0-9]*\.?[0-9]*)%\s*Packaging & Ship"),
    # percentage with optional decimal point -> newline -> "End-of-Life Management"
    re.compile(r"(?P<gwp_eol_ratio>[0-9]*\.?[0-9]*)%\s*End-of-Life Management"),
    # percentage with optional decimal point -> newline -> "Manufacturing"
    re.compile(r"(?P<gwp_manufacturing_ratio>[0-9]*\.?[0-9]*)%\s*Manufacturing"),
    # percentage with optional decimal point -> "PCB"
    re.compile(r"(?P<pcb_ratio>[0-9]*\.?[0-9]*)%\s*PCB"),
    # percentage with optional decimal point -> "LCD"
    re.compile(r"(?P<lcd_ratio>[0-9]*\.?[0-9]*)%\s*LCD"),
    # percentage with optional decimal point -> "System component"
    re.compile(r"(?P<system_component_ratio>[0-9]*\.?[0-9]*)%\s*System component"),
    # percentage with optional decimal point -> "Inductor"
    re.compile(r"(?P<inductor_ratio>[0-9]*\.?[0-9]*)%\s*Inductor"),
    # percentage with optional decimal point -> "Keyboard"
    re.compile(r"(?P<keyboard_ratio>[0-9]*\.?[0-9]*)%\s*Keyboard"),
    # percentage with optional decimal point -> "Storage"
    re.compile(r"(?P<storage_ratio>[0-9]*\.?[0-9]*)%\s*Storage"),
    # percentage with optional decimal point -> "IC"
    re.compile(r"(?P<ic_ratio>[0-9]*\.?[0-9]*)%\s*IC"),
    # use 90% \n recycled content by total weight of
    re.compile(r"use\s*(?P<recycled_packaging>[0-9]*\.?[0-9]*)%\s*recycled content"),
    # 18.18kwh -> "Use energy" -> "demand (Yearly TEC)""
    re.compile(r"(?P<use_energy>[0-9]*\.?[0-9]*)kwh.*Use energy demand \(Year TEC\)"),
    re.compile(r"Etec\s*\(KWh/year\)\s*(?P<use_energy>[0-9]*\.?[0-9]*)"),
    # Product modular design, 90% materials and components are -> easy to recycle and reuse.
    re.compile(
        r"Product modular design, (?P<recycled_components>[0-9]*\.?[0-9]*)%\s*materials and components are"
    ),
)


def asus_parse(body, filename):
    result = data.DeviceCarbonFootprintData()

    # Parse text from PDF.
    pages = []
    for page in range(pymupdf.open("pdf", body).page_count):
        pdf_as_image = pdf.pdf2img(body, page)
        pages.append(pytesseract.image_to_string(pdf_as_image))
    pdf_as_text = "\n".join(pages)
    extracted = text.search_all_patterns(_ASUS_LCA_PATTERNS, pdf_as_text)
    if not extracted:
        return

    # Convert each matched group to boavizta format.
    if "gwp_manufacturing_ratio" in extracted:
        result["gwp_manufacturing_ratio"] = round(
            float(extracted["gwp_manufacturing_ratio"]) / 100, 3
        )
    if "gwp_use_ratio" in extracted:
        result["gwp_use_ratio"] = round(float(extracted["gwp_use_ratio"]) / 100, 3)
    if "gwp_eol_ratio" in extracted:
        result["gwp_eol_ratio"] = round(float(extracted["gwp_eol_ratio"]) / 100, 3)
    if "gwp_transport_ratio" in extracted:
        result["gwp_transport_ratio"] = float(extracted["gwp_transport_ratio"]) / 100
    if "pcb_ratio" in extracted:
        result["pcb_ratio"] = float(extracted["pcb_ratio"]) / 100  # type: ignore
    if "lcd_ratio" in extracted:
        result["lcd_ratio"] = float(extracted["lcd_ratio"]) / 100  # type: ignore
    if "system_component_ratio" in extracted:
        result["system_component_ratio"] = (  # type: ignore
            float(extracted["system_component_ratio"]) / 100
        )
    if "inductor_ratio" in extracted:
        result["inductor_ratio"] = float(extracted["inductor_ratio"]) / 100  # type: ignore
    if "keyboard_ratio" in extracted:
        result["keyboard_ratio"] = float(extracted["keyboard_ratio"]) / 100  # type: ignore
    if "storage_ratio" in extracted:
        result["storage_ratio"] = float(extracted["storage_ratio"]) / 100  # type: ignore
    if "ic_ratio" in extracted:
        result["ic_ratio"] = float(extracted["ic_ratio"]) / 100  # type: ignore
    if "use_location" in extracted:
        result["use_location"] = extracted["use_location"]
    if not "category" in result:
        result["category"] = "Workplace"
    if "date" in extracted:
        result["report_date"] = extracted["date"]
    if "screen_size" in extracted:
        result["screen_size"] = float(extracted["screen_size"])
    if "assembly_location" in extracted:
        result["assembly_location"] = extracted["assembly_location"]
    if "lifetime" in extracted:
        result["lifetime"] = int(extracted["lifetime"])
    if "name" in extracted:
        result["name"] = extracted["name"]
    if "weight" in extracted:
        result["weight"] = float(extracted["weight"])
    if "use_energy" in extracted:
        result["yearly_tec"] = float(extracted["use_energy"])
    if "footprint" in extracted:
        result["gwp_total"] = float(extracted["footprint"])

    now = datetime.datetime.now()
    result["added_date"] = now.strftime("%Y-%m-%d")
    result["add_method"] = "Asus Auto Parser"
    result["manufacturer"] = "ASUS"
    yield data.DeviceCarbonFootprint(result)
