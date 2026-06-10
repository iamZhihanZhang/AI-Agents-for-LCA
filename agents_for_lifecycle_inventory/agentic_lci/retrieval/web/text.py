# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Fall back web search if the LLM provider does not natively support it

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import os
import requests
import tempfile

import html_to_markdown
import pymupdf4llm

from .util import USER_AGENT

MAX_PDF_FILE_BYTES = 2_000_000

from dotenv import load_dotenv

load_dotenv()

assert os.environ.get(
    "GOOGLE_SEARCH_API_KEY"
), "Please add a Google Search API key to the .env file"
assert os.environ.get(
    "GOOGLE_SEARCH_ENGINE_ID"
), "Please add a Google Search engine id to the .env file"


def google_search_for_matching_urls(query: str, num_results=10):
    search_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={os.environ['GOOGLE_SEARCH_API_KEY']}&cx={os.environ['GOOGLE_SEARCH_ENGINE_ID']}&num={num_results}"
    response = requests.get(search_url)
    response.raise_for_status()
    search_results = response.json()

    search_data = "\n".join(
        [f"{item['link']} {item['snippet']}" for item in search_results["items"]]
    )
    return search_data


def url_to_markdown(url: str) -> str:
    response = requests.get(url, headers=USER_AGENT)
    response.raise_for_status()

    if ".pdf" in url:
        with tempfile.NamedTemporaryFile() as tmp:
            tmp.write(response.content)
            tmp.seek(0)
            if len(tmp.read()) > MAX_PDF_FILE_BYTES:
                raise ValueError("File too large")
            tmp.seek(0)
            return pymupdf4llm.to_markdown(tmp.name)  # type: ignore
    else:
        return str(
            html_to_markdown.convert(response.content.decode("utf-8", errors="ignore"))
        )


if __name__ == "__main__":
    # example/test usage
    print(google_search_for_matching_urls("Koel Labs", num_results=1))
    print(url_to_markdown("https://sandergi.com"))
