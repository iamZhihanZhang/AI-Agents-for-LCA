# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Geolocation (useful for figuring out location which is relevant to electricity footprint etc.)

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import requests
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="LCA-Server")


def lookup_location_name(lat: float, lng: float) -> str:
    """
    Get the location name in 'City, State' format from latitude and longitude.
    """
    location = geolocator.reverse(f"{lat}, {lng}")
    return location.address if location else "Seattle, WA"  # type: ignore


def lookup_coordinates(location_name: str) -> dict[str, float]:
    """
    Get the latitude and longitude from a location name. Returns a dictionary with 'lat' and 'lng' keys.
    """
    location = geolocator.geocode(location_name)
    return {"lat": location.latitude, "lng": location.longitude}  # type: ignore


def lookup_current_location(ip: str | None = None) -> dict[str, float | str]:
    """
    Get the current location of the user using IP address. Defaults to the server location if IP is not provided.
    Returns a dictionary with 'lat', 'lng', and 'name' keys. The name is in 'City, State' format.
    """
    if ip:
        response = requests.get(f"https://ipinfo.io/{ip}")
    else:
        response = requests.get("https://ipinfo.io")
    if response.ok:
        data = response.json()
        return {
            "lat": float(data["loc"].split(",")[0]),
            "lng": float(data["loc"].split(",")[1]),
            "name": data["city"] + ", " + data["region"],
        }
    return {"lat": 47.6062, "lng": -122.3321, "name": "Seattle, WA"}
