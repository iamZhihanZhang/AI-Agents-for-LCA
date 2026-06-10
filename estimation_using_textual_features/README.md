# Weighted k-Nearest Neighbor Estimator

A weighted k-NN framework for environmental impact estimation that handles **heterogeneous attribute availability**, **mixed numerical/categorical features**, and provides **Gaussian uncertainty estimates** (point prediction + 95% confidence interval).

Designed for Life Cycle Assessment (LCA) CO2e prediction, but generalizes to any regression task with incomplete, mixed-type features.

## Features

- **NaN-aware distance computation** — compares products even when they have different available attributes, using only mutually observed features
- **Mixed data types** — numerical features are z-score normalized; categorical features are one-hot encoded; both preserve NaN for missing attributes
- **Six weighting strategies** — uniform, inverse-distance, completeness, shared-features, and two hybrid schemes
- **Gaussian uncertainty** — returns weighted mean ± 95% confidence interval from neighbor distribution
- **Pluggable data loaders** — built-in support for electricity grid data and product CSV files with per-category attribute schemas

## Installation

```bash
pip install numpy pandas scikit-learn scipy
```

## Quick Start

```python
from sklearn.model_selection import train_test_split
from estimation_using_textual_features import WeightedKNNEstimator, WeightStrategy
from estimation_using_textual_features.data_loaders import load_electricity_data

# Load and split data
df = load_electricity_data("data/ElectricityMaps.txt", include_zone=False)
train, test = train_test_split(df, test_size=0.2, random_state=42)

# Train and predict
model = WeightedKNNEstimator(k=5, weight_strategy=WeightStrategy.COMPLETENESS)
model.fit(train)
results = model.predict(test)

# Inspect results
print(results.summary())
print(results.to_dataframe().head())
```

## Weighting Strategies

| Strategy | Description |
|---|---|
| `UNIFORM` | Equal weight for all k neighbors |
| `INVERSE_DISTANCE` | Weight ∝ 1/distance |
| `COMPLETENESS` | Weight = number of observed attributes per neighbor |
| `SHARED_FEATURES` | Weight = number of attributes co-observed with query |
| `HYBRID_COMPLETENESS` | (1/distance) × completeness |
| `HYBRID_SHARED` | (1/distance) × shared features |

## Running Experiments

The `run_experiments.py` script runs the full evaluation pipeline:

```bash
# Electricity data
python -m estimation_using_textual_features.run_experiments --data data/ElectricityMaps.txt

# Product CSV data
python -m estimation_using_textual_features.run_experiments --data data/products.csv --product-class Laptop
```

This produces CSV files in `results/` for:
- Data scaling analysis (varying training size × k)
- Synthetic missingness robustness
- Runtime efficiency benchmarking
- Baseline comparisons (sklearn SVM and kNN)
- 5-fold cross-validation

## Project Structure

```
estimation_using_textual_features/
├── __init__.py           # Public API exports
├── weighted_knn_estimator.py          # Core WeightedKNNEstimator, FeaturePreprocessor
├── data_loaders.py       # Data format handlers (electricity, product CSV)
├── evaluation.py         # Experiment utilities (scaling, missingness, etc.)
└── run_experiments.py    # CLI entry point for full pipeline
```

## Loading Custom Data

### Electricity data (line-delimited dicts)

```python
from estimation_using_textual_features.data_loaders import load_electricity_data
df = load_electricity_data("data/ElectricityMaps.txt")
```

### Product CSV with per-category schemas

```python
from estimation_using_textual_features.data_loaders import load_product_csv
df, num_cols, cat_cols = load_product_csv(
    "products.csv",
    product_class="Laptop",
    target_column="gwp_total_kg_co2e",
)
```

You can pass custom schemas and feature lists to `load_product_csv` to support new product categories.

## Citation

If you use this code, please cite the accompanying paper.
