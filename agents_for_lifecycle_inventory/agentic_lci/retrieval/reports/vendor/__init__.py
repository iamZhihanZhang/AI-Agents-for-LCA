# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Vendor specific report parsers

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import requests
from io import BytesIO

from agentic_lci.retrieval.web.util import USER_AGENT

from .asus import asus_parse
from .boavizta.apple import parse as apple_parse
from .boavizta.dell_laptop import parse as dell_parse
from .boavizta.google_parse import parse as google_parse
from .boavizta.hp_workplace import parse as hp_parse
from .boavizta.hpe import parse as hpe_parse
from .boavizta.huawei import parse as huawei_parse
from .boavizta.lenovo import parse as lenovo_parse
from .boavizta.microsoft import parse as microsoft_parse


def from_report(report, url):
    from agentic_lci.schemas.electronics_gwp import from_report as _from_report

    return _from_report(report, url)


def parse_asus(url):
    bytz = BytesIO()
    bytz.write(requests.get(url, headers=USER_AGENT).content)
    bytz.seek(0)
    if len(bytz.getvalue()) > 2_000_000:
        raise ValueError("File too large")
    bytz.seek(0)
    res = next(asus_parse(bytz, url))
    return from_report(res, url)


def parse_dell(url):
    bytz = BytesIO()
    bytz.write(requests.get(url, headers=USER_AGENT).content)
    bytz.seek(0)
    if len(bytz.getvalue()) > 2_000_000:
        raise ValueError("File too large")
    bytz.seek(0)
    res = next(dell_parse(bytz, url))
    return from_report(res, url)


def parse_apple(url):
    bytz = BytesIO()
    bytz.write(requests.get(url, headers=USER_AGENT).content)
    bytz.seek(0)
    if len(bytz.getvalue()) > 2_000_000:
        raise ValueError("File too large")
    bytz.seek(0)
    res = next(apple_parse(bytz, url))
    return from_report(res, url)


def parse_google(url):
    bytz = BytesIO()
    bytz.write(requests.get(url, headers=USER_AGENT).content)
    bytz.seek(0)
    if len(bytz.getvalue()) > 2_000_000:
        raise ValueError("File too large")
    bytz.seek(0)
    res = next(google_parse(bytz, url))
    return from_report(res, url)


def parse_hp(url):
    bytz = BytesIO()
    bytz.write(requests.get(url, headers=USER_AGENT).content)
    bytz.seek(0)
    if len(bytz.getvalue()) > 2_000_000:
        raise ValueError("File too large")
    bytz.seek(0)
    res = next(hp_parse(bytz, url))
    return from_report(res, url)


def parse_hpe(url):
    bytz = BytesIO()
    bytz.write(requests.get(url, headers=USER_AGENT).content)
    bytz.seek(0)
    if len(bytz.getvalue()) > 2_000_000:
        raise ValueError("File too large")
    bytz.seek(0)
    res = next(hpe_parse(bytz, url))
    return from_report(res, url)


def parse_huawei(url):
    bytz = BytesIO()
    bytz.write(requests.get(url, headers=USER_AGENT).content)
    bytz.seek(0)
    if len(bytz.getvalue()) > 2_000_000:
        raise ValueError("File too large")
    bytz.seek(0)
    res = next(huawei_parse(bytz, url))
    return from_report(res, url)


def parse_lenovo(url):
    bytz = BytesIO()
    bytz.write(requests.get(url, headers=USER_AGENT).content)
    bytz.seek(0)
    if len(bytz.getvalue()) > 2_000_000:
        raise ValueError("File too large")
    bytz.seek(0)
    res = next(lenovo_parse(bytz, url))
    return from_report(res, url)


def parse_microsoft(url):
    bytz = BytesIO()
    bytz.write(requests.get(url, headers=USER_AGENT).content)
    bytz.seek(0)
    if len(bytz.getvalue()) > 2_000_000:
        raise ValueError("File too large")
    bytz.seek(0)
    res = next(microsoft_parse(bytz, url))
    return from_report(res, url)
