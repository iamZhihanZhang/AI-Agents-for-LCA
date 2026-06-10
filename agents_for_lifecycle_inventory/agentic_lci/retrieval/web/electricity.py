# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Carbon Intensity API for Electricity Usage

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import os
import requests

assert os.environ[
    "ELECTRICITYMAPS_API_KEY"
], "Please add an electricitymaps api key to the .env"


def lookup_electricity_carbon_intensity(lat: float, lng: float) -> float:
    response = requests.get(
        f"https://api.electricitymap.org/v3/carbon-intensity/history?lat={lat}&lon={lng}",
        headers={"auth-token": os.environ["ELECTRICITYMAPS_API_KEY"]},
    )
    intensity = 24
    if response.ok:
        data = response.json()
        total = 0
        for datapoint in data["history"]:
            if not datapoint["carbonIntensity"]:
                continue
            total += datapoint["carbonIntensity"]
        intensity = total / len(data["history"])
        intensity = intensity or 24  # default to 24 if no data
    else:
        # Failed to get carbon intensity
        pass
    return intensity
