# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Main entry point for the unified tool layer using the standardized Model Context Protocol (MCP)

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import warnings

warnings.filterwarnings("ignore")

import logging
import sys

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
)

import os
import cv2
import requests
import numpy as np
from PIL import Image
from io import BytesIO

dirname = os.path.dirname(__file__)

# Initialize FastMCP server
from mcp.server.fastmcp import FastMCP, Image as MCPImage
from mcp.types import ImageContent

mcp = FastMCP(
    name="Life Cycle Assessment",
    instructions="""
    These are a set of tools to help with retrieving information to complete life cycle inventories (LCIs) for life cycle assessment (LCA).
    """,
)


# Utility functions
def standardize_product_name(product_name: str):
    return product_name.lower().replace(" ", "").removeprefix("asus")


def encode_image(image: Image.Image) -> ImageContent:
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_bytes = buffer.getvalue()
    img_obj = MCPImage(data=img_bytes, format="png")
    return img_obj.to_image_content()


# ============================== Database Lookup ==============================
from agentic_lci.retrieval.database.electronics import combined_df


@mcp.tool()
def lookup_db_info(product_name: str) -> dict | None:
    """
    Look up the product name in the database. If no match is found, return None.
    """
    matching_rows = combined_df[
        combined_df["product_name"].apply(
            lambda s: standardize_product_name(s).startswith(
                standardize_product_name(product_name)
            )
        )
    ]
    if len(matching_rows) <= 0:
        return None
    return matching_rows.to_dict()


# ============================== Environmental Report Parsing ==============================
from agentic_lci.retrieval.reports import parse_report, ProductReportModel


@mcp.tool()
def attempt_environmental_report_parse(
    candidate_env_report_urls: set,
) -> ProductReportModel | None:
    """
    Attempt to parse the environmental report from the given URLs using a purpose-built parser.
    Returns the first successful parse result or None if none of the URLs match an expected format.
    If this doesn't work, you may have to read the PDF yourself. It is recommended to sanity check
    the output (e.g., cross reference already known values like the product name) to make sure the
    right report was found and that parsing was indeed successful.
    """

    res = None
    for url in candidate_env_report_urls:
        if "guide" in url or "manual" in url:
            continue
        res = parse_report(url)
        if res:
            break

    return res


# ============================== Search ==============================
from agentic_lci.retrieval.web.text import (
    google_search_for_matching_urls,
    url_to_markdown,
)
from agentic_lci.retrieval.web.image import google_image_search


@mcp.tool()
def search_google_for_matching_urls(query: str):
    """
    Find urls to PDFs and Web Pages relevant to the query.
    Includes a snippet from each found web resource.
    Some useful queries include "environmental report pdf",
    "carbon footprint report pdf", "pcf datasheet pdf",
    "component list", "teardown", etc. with the product name/identifiers
    """
    return google_search_for_matching_urls(query)


@mcp.tool()
def search_google_for_matching_image_urls(query: str):
    """
    Find urls to images and photos relevant to the query.
    Useful to find internal PCB/product photos (a good
    search query here is "chip id" followed by the id of the chip if known)
    """
    return [encode_image(i) for i in google_image_search(query)]


@mcp.tool()
def convert_url_to_markdown(url: str):
    """
    Converts a PDF/HTML page into readable markdown.
    """
    return url_to_markdown(url)


# ============================== Usage Phase ==============================
from agentic_lci.retrieval.web.electricity import lookup_electricity_carbon_intensity
from agentic_lci.retrieval.web.geo import (
    lookup_coordinates,
    lookup_current_location,
    lookup_location_name,
)


@mcp.tool()
def get_location_name(lat: float, lng: float) -> str:
    """
    Get the location name in 'City, State' format from latitude and longitude.
    """
    return lookup_location_name(lat, lng)


@mcp.tool()
def get_coordinates(location_name: str) -> dict[str, float]:
    """
    Get the latitude and longitude from a location name. Returns a dictionary with 'lat' and 'lng' keys.
    """
    return lookup_coordinates(location_name)


@mcp.tool()
def get_current_location(ip: str | None = None) -> dict[str, float | str]:
    """
    Get the current location of the user using IP address. Defaults to the server location if IP is not provided.
    Returns a dictionary with 'lat', 'lng', and 'name' keys. The name is in 'City, State' format.
    """
    return lookup_current_location(ip)


@mcp.tool()
def get_electricity_carbon_intensity(lat: float, lng: float) -> float:
    """
    Get the carbon intensity of electricity in grams of CO2 per kWh for a given latitude and longitude.
    You can multiply the expected lifetime of a product by the yearly energy consumption and this value to estimate the total carbon footprint of the usage phase.
    """
    return lookup_electricity_carbon_intensity(lat, lng)


# ============================== Vision Pipeline ==============================
from agentic_lci.retrieval.web.image import lookup_fcc_photos
from agentic_lci.retrieval.vision import (
    extract_components_and_dimensions_from_internal_photos,
)


@mcp.tool()
def scrape_fcc_photos(fcc_id: str):
    """
    Given an FCC ID, this retrieves the internal product photos from the FCC report(s) if present.
    These are nice since they have rulers which make optaining dimension information easier.
    If this fails to find any usable photos, you will have to find internal product photos via regular web search
    from product tear downs.
    """
    return [encode_image(i) for i in lookup_fcc_photos(fcc_id)]


@mcp.tool()
def calculate_components_and_dimensions_from_internal_photos(image_urls: list[str]):
    """
    USEFUL to find the dimensions of components for the product.
    Attempts to detect which of the image_urls contain the PCB, tries to find rulers in the image, then tries to read the ruler to determine the dimensions
    of the electronic components present in the image.

    The result is a dictionary with the following keys:
        - component_descriptions: the types of detected components and pcbs including dimensions (you might need to reason through how to pair this with data you find online)
        - image_with_pcb_highlighted: the PCB is highlighted so you can confirm that it was correctly detected and ignore the results otherwise
        - just_ruler_image: an image cropped to just the detected ruler so you can confirm it was detected correctly
        - components_image: an image cropped to the PCB with bounding boxes labeled for the components

    If it seems the ruler is incorrectly detected, you can try to read it yourself and consult the bounding boxes.
    If it seems there is no ruler, you can identify a common component which has a known size and use that to estimate the size of the other components.
    """
    images = []
    for url in image_urls:
        try:
            response = requests.get(url)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content))
            img = np.array(img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            images.append(img)
        except requests.exceptions.RequestException as e:
            return f"Error fetching image: {e}"

    (
        component_descriptions,
        image_with_pcb_highlighted,
        just_ruler_image,
        components_image,
    ) = extract_components_and_dimensions_from_internal_photos(images)
    return {
        "component_descriptions": component_descriptions,
        "image_with_pcb_highlighted": encode_image(
            Image.fromarray(cv2.cvtColor(image_with_pcb_highlighted, cv2.COLOR_BGR2RGB))
        ),
        "just_ruler_image": encode_image(
            Image.fromarray(cv2.cvtColor(just_ruler_image, cv2.COLOR_BGR2RGB))
        ),
        "components_image": encode_image(
            Image.fromarray(cv2.cvtColor(components_image, cv2.COLOR_BGR2RGB))
        ),
    }


# ============================== Initialize and run the server ==============================
if __name__ == "__main__":
    mcp.run("streamable-http")
