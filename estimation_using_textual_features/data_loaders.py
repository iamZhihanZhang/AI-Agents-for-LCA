"""
Data loaders for the Weighted kNN Estimator.

Each loader reads a specific raw data format and returns a clean
``pandas.DataFrame`` ready for use with ``WeightedKNNEstimator``.

Supported formats:
    - Electricity carbon-intensity data (line-delimited Python dicts)
    - Product LCA data from CSV with per-category attribute schemas

Usage::

    from data_loaders import load_electricity_data, load_product_csv

    df_elec = load_electricity_data("data/ElectricityMaps.txt")
    df_prod = load_product_csv("data/products.csv", product_class="Laptop")
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Electricity data
# ---------------------------------------------------------------------------

def load_electricity_data(
    filepath: str | Path,
    target_column: str = "carbon_intensity",
    include_zone: bool = True,
) -> pd.DataFrame:
    """Load electricity carbon-intensity data from a line-delimited text file.

    Each line is expected to be a Python dict literal with at least the keys
    ``powerConsumptionBreakdown``, ``carbonIntensity``, and optionally
    ``powerImportTotal`` and ``zone``.

    Power source values are normalised to fractions of total power
    (consumption + imports) so that each row sums to ~1.

    Parameters:
        filepath: Path to the ``ElectricityMaps.txt`` file.
        target_column: Name to use for the target column in the output
            DataFrame (default ``"carbon_intensity"``).
        include_zone: If *True*, retain the ``zone`` column for
            stratified analysis or cross-validation.

    Returns:
        DataFrame with one row per valid record.  Feature columns are the
        normalised power-source fractions (+ ``power_import``).  The target
        column contains the raw ``carbonIntensity`` value.
    """
    filepath = Path(filepath)
    rows: list[dict] = []

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or "Power breakdown request failed" in line:
                continue

            try:
                record = ast.literal_eval(line)
            except (SyntaxError, ValueError):
                continue

            power = record.get("powerConsumptionBreakdown", {})
            import_total = record.get("powerImportTotal", 0) or 0
            carbon = record.get("carbonIntensity")

            if not power or carbon is None:
                continue

            total = sum(power.values()) + import_total
            if total == 0:
                continue

            row = {src: val / total for src, val in power.items()}
            row["power_import"] = import_total / total
            row[target_column] = carbon

            if include_zone:
                row["zone"] = record.get("zone")

            rows.append(row)

    df = pd.DataFrame(rows)

    # Fill missing power sources with 0 (some zones lack certain sources)
    feature_cols = [
        c for c in df.columns if c not in (target_column, "zone")
    ]
    df[feature_cols] = df[feature_cols].fillna(0.0)

    return df


# ---------------------------------------------------------------------------
# Product CSV data with per-category schemas
# ---------------------------------------------------------------------------

# Default attribute schemas per product class.  Users can extend or override.
DEFAULT_PRODUCT_SCHEMA: dict[str, list[str]] = {
    'Desktop': [
        # CPU
        'cpu_brand', 'cpu_generation', 'number_of_cores_CPU',
        # GPU
        'gpu_brand', 'gpu_series_number',
        # Memory & Storage
        'memory_gb', 'hard_drive_type', 'hard_drive_size_gb',
        # Physical
        'dimension_volume_cm3', 'weight_kg', 
        # Energy
        'lifetime_years', 'TEC of Model (kWh)', 'yearly_household_energy_consumption_kWh',
        # Certification
        'Climate+', 'epeat_score', 'EPEAT Tier', 'TCO Certified'
    ],
    'Laptop': [
        # CPU
        'cpu_brand', 'cpu_generation', 'number_of_cores_CPU',
        # GPU
        'gpu_brand', 'gpu_series_number',
        # Memory & Storage
        'memory_gb', 'hard_drive_type', 'hard_drive_size_gb',
        # Display
        'display_size', 'display_type', 'display_max_brightness_nits', 'display_refresh_rate_hz',
        # Battery
        'battery_capacity_mah',
        # Physical
        'dimension_volume_cm3', 'weight_kg', 
        # Energy
        'lifetime_years', 'TEC of Model (kWh)', 'yearly_household_energy_consumption_kWh',
        # Certification
        'Climate+', 'epeat_score', 'EPEAT Tier', 'TCO Certified'
    ],
    'Display': [
        # Display
        'display_size', 'display_type', 'display_max_brightness_nits', 'display_refresh_rate_hz',
        'Panel Type', 'native_res_prod', 'Screen Size (inches)', 'Screen Area (square inches)',
        # Physical
        'dimension_volume_cm3', 'weight_kg', 
        # Energy
        'lifetime_years', 'TEC of Model (kWh)', 'yearly_household_energy_consumption_kWh',
        'On Mode Power (watts)', 'Sleep Mode Power (watts)',
        # Certification
        'Climate+', 'epeat_score', 'EPEAT Tier', 'TCO Certified'
    ],
}

# Default lists for type disambiguation
DEFAULT_NUMERIC_FEATURES = [
    # --- Processor ---
    'cpu_generation',           # e.g., 12th gen, 13th gen
    'number_of_cores_CPU',      # CPU Core count
    'gpu_series_number',        # e.g., 3000, 4000 series
    'number_of_cores_GPU',      # GPU Core count
    'chip_SoC',                 # Binary/Categorical coded as int often
    'number_of_cores_SoC',      # System-on-Chip Core count

    # --- Memory & Storage ---
    'memory_gb',                # RAM size
    'hard_drive_size_gb',       # SSD/HDD storage capacity

    # --- Display & Screen ---
    'display_size',             # Diagonal size (generic)
    'display_max_brightness_nits', # Peak brightness
    'display_refresh_rate_hz',     # Refresh rate (60Hz, 144Hz, etc.)
    'native_res_prod',             # Resolution product (Width x Height)
    'Screen Size (inches)',     # Diagonal size (specific)
    'Screen Area (square inches)', # Total area

    # --- Power & Battery ---
    'battery_capacity_mah',     # Battery size
    'TEC of Model (kWh)',       # Typical Energy Consumption
    'yearly_household_energy_consumption_kWh', # Est. yearly energy
    'On Mode Power (watts)',    # Active power draw
    'Sleep Mode Power (watts)', # Standby power draw

    # --- Physical Dimensions ---
    'dimension_width_cm',
    'dimension_height_cm',
    'dimension_depth_cm',
    'dimension_volume_cm3',     # Calculated volume
    'weight_kg',

    # --- Sustainability Metrics ---
    'lifetime_years',           # Expected use-life
    'epeat_score',              # EPEAT certification score
]

DEFAULT_CATEGORICAL_FEATURES = [
    "cpu_brand", "gpu_brand", "Climate+", "EPEAT Tier", "TCO Certified",
    "hard_drive_type", "display_type", "Panel Type",
]


def load_product_csv(
    filepath: str | Path,
    product_class: str,
    target_column: str = "carbon_intensity",
    schema: dict[str, list[str]] | None = None,
    numeric_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """Load a product CSV and select columns for a given product class.

    Parameters:
        filepath: Path to the CSV file.
        product_class: Key into the schema dict (e.g. ``"Laptop"``).
        target_column: Name of the target column in the CSV.
        schema: Mapping from product class to list of feature column names.
            Defaults to ``DEFAULT_PRODUCT_SCHEMA``.
        numeric_features: Override list of numerical feature names.
        categorical_features: Override list of categorical feature names.

    Returns:
        A tuple of ``(df, numeric_cols, categorical_cols)`` where *df*
        contains only the schema-selected columns plus the target, and
        the two lists classify each selected column as numerical or
        categorical.
    """
    schema = schema or DEFAULT_PRODUCT_SCHEMA
    if product_class not in schema:
        raise ValueError(
            f"Unknown product class '{product_class}'. "
            f"Available: {list(schema.keys())}"
        )

    raw = pd.read_csv(filepath)
    if target_column not in raw.columns:
        raise ValueError(
            f"Target column '{target_column}' not found in CSV. "
            f"Available columns: {list(raw.columns)}"
        )

    # Select columns present in both the schema and the CSV
    desired = schema[product_class]
    available = [c for c in desired if c in raw.columns]

    # Classify into numeric / categorical
    num_ref = set(numeric_features or DEFAULT_NUMERIC_FEATURES)
    cat_ref = set(categorical_features or DEFAULT_CATEGORICAL_FEATURES)

    num_cols = [c for c in available if c in num_ref]
    cat_cols = [c for c in available if c in cat_ref]

    # Keep only classified columns + target
    keep = num_cols + cat_cols + [target_column]
    df = raw[keep].copy()

    return df, num_cols, cat_cols
