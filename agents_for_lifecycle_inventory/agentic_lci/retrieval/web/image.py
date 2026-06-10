# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Image search utilities

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import os
import tempfile
import requests
from io import BytesIO
from PIL import Image

import pymupdf
from bs4 import BeautifulSoup

from .util import USER_AGENT

from dotenv import load_dotenv

load_dotenv()

assert os.environ.get(
    "GOOGLE_SEARCH_API_KEY"
), "Please add a Google Search API key to the .env file"
assert os.environ.get(
    "GOOGLE_SEARCH_ENGINE_ID"
), "Please add a Google Search engine id to the .env file"


def google_image_search(query: str, num_results=10) -> list[Image.Image]:
    response = requests.get(
        f"https://www.googleapis.com/customsearch/v1?key={os.environ['GOOGLE_SEARCH_API_KEY']}&cx={os.environ['GOOGLE_SEARCH_ENGINE_ID']}&q={query}&imgType=photo&searchType=image&num={num_results}"
    )
    response.raise_for_status()
    data = response.json()

    images = []
    for item in data["items"]:
        image_bytes = BytesIO(requests.get(item["link"]).content)
        image = Image.open(image_bytes)
        images.append(image)
    return images


def lookup_fcc_photos(fcc_id: str) -> list[Image.Image]:
    # access the table row(s) from the FCC websites with the "internal photos" label
    response = requests.get(f"https://fcc.report/FCC-ID/{fcc_id}", headers=USER_AGENT)
    response.raise_for_status()
    html = response.text
    soup = BeautifulSoup(html, "html.parser")

    # find table rows with a cell containing "Internal Photos" case insensitive
    links = []
    rows = soup.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        for cell in cells:
            if "internal photos" in cell.text.lower():
                # identify the link tag in the row
                link = row.find("a")
                if link:
                    internal_photos_link = f'https://fcc.report{link["href"]}'
                    links.append(internal_photos_link)
                break

    # extract the images from the pdf linked on the internal photos page
    images = []
    for link in links:
        response = requests.get(f"{link}.pdf")
        response.raise_for_status()
        with tempfile.NamedTemporaryFile(suffix=".pdf") as f:
            f.write(response.content)

            pdf_file = pymupdf.open(f.name)

            # iterate over PDF pages
            for page_index in range(len(pdf_file)):
                page = pdf_file.load_page(page_index)
                image_list = page.get_images()

                for img in image_list:

                    # get the XREF of the image
                    xref = img[0]

                    # extract the image bytes
                    base_image = pdf_file.extract_image(xref)
                    image_bytes = base_image["image"]

                    # get the image extension
                    image_ext = base_image["ext"]

                    # load the image
                    with tempfile.NamedTemporaryFile(suffix=f".{image_ext}") as f:
                        f.write(image_bytes)
                        image_path = f.name
                        try:
                            images.append(Image.open(image_path))
                        except Exception as e:
                            print(f"Error reading image: {e}")

    return images


if __name__ == "__main__":
    # example/test usage
    google_image_search("Koel Labs", num_results=1)[0].save("temp1.png")
    lookup_fcc_photos("BCG-E3997A")[0].save("temp2.png")
