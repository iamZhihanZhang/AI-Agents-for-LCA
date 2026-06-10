# SPDX-License-Identifier: LicenseRef-AI-Agents-for-LCA-Noncommercial
"""
Agentic LCA - Database lookup of pre-parsed and verified electronic product information

Part of the codebase accompanying:
"Sustainability assessment using multimodal artificial intelligence agents"
Nature Electronics (2026)

See the LICENSE file in the repository root for details.
"""

import os
import pandas as pd

data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data")

apple_df = pd.read_csv(os.path.join(data_dir, "apple.csv"))
# df_2 = pd.read_csv("") # Placeholder for additional datasets, e.g., from other PCF databases like https://github.com/Boavizta/environmental-footprint-data/blob/main/boavizta-data-us.csv

# Columns: ['product_class', 'product_name', 'manufacturer', 'report_date', 'gwp_total_kg_co2e']
surface_9 = {
    "product_class": "Laptop",
    "product_name": "Surface Pro 9 5G",
    "manufacturer": "Microsoft",
    "report_date": None,
    "gwp_total_kg_co2e": 196.0,
}
combined_df = pd.concat([pd.DataFrame([surface_9]), apple_df], ignore_index=True)
