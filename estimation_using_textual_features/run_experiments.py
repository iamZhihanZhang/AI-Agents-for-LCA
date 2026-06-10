#!/usr/bin/env python3
"""
Example: Full evaluation pipeline for the Weighted kNN Estimator.

This script demonstrates how to:
    1. Load data (electricity or product CSV)
    2. Train and evaluate the weighted kNN estimator
    3. Run data scaling analysis
    4. Run synthetic missingness evaluation
    5. Benchmark runtime efficiency
    6. Compare against sklearn baselines
    7. Run cross-validation

Usage:
    python run_experiments.py --data data/ElectricityMaps.txt
    python run_experiments.py --data data/products.csv --product-class Laptop
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from weighted_knn_estimator  import WeightedKNNEstimator, WeightStrategy
from data_loaders import load_electricity_data, load_product_csv
from evaluation import (
    cross_validate,
    evaluate_baselines,
    evaluate_data_scaling,
    evaluate_missingness,
    evaluate_runtime,
)


def main():
    parser = argparse.ArgumentParser(
        description="Run weighted kNN experiments."
    )
    parser.add_argument(
        "--data", type=str, required=True,
        help="Path to data file (.txt for electricity, .csv for products).",
    )
    parser.add_argument(
        "--product-class", type=str, default=None,
        help="Product class for CSV data (e.g. 'Laptop', 'Desktop').",
    )
    parser.add_argument(
        "--target", type=str, default="carbon_intensity",
        help="Name of the target column.",
    )
    parser.add_argument(
        "--output-dir", type=str, default="results",
        help="Directory to save result CSVs.",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2,
        help="Fraction of data to hold out for testing.",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for train/test split.",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Load data --------------------------------------------------------
    data_path = Path(args.data)
    num_feats = None
    cat_feats = None

    if data_path.suffix == ".txt":
        print(f"Loading electricity data from {data_path} ...")
        df = load_electricity_data(data_path, target_column=args.target, include_zone=False)
    elif data_path.suffix == ".csv":
        if args.product_class is None:
            parser.error("--product-class is required for CSV data.")
        print(f"Loading {args.product_class} data from {data_path} ...")
        df, num_feats, cat_feats = load_product_csv(
            data_path, args.product_class, target_column=args.target,
        )
    else:
        raise ValueError(f"Unsupported file extension: {data_path.suffix}")

    print(f"  Loaded {len(df)} rows, {len(df.columns)} columns.")

    train_df, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed
    )
    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    print(f"  Train: {len(train_df)}, Test: {len(test_df)}")

    # ---- 1. Quick evaluation ----------------------------------------------
    print("\n=== Quick Evaluation (k=5, completeness weighting) ===")
    model = WeightedKNNEstimator(
        k=5,
        weight_strategy=WeightStrategy.COMPLETENESS,
        target_column=args.target,
        numerical_features=num_feats,
        categorical_features=cat_feats,
    )
    model.fit(train_df)
    results = model.predict(test_df)
    for key, val in results.summary().items():
        print(f"  {key}: {val}")
    results.to_dataframe().to_csv(out_dir / "quick_eval.csv", index=False)

    # ---- 2. Data Scaling Analysis -----------------------------------------
    print("\n=== Data Scaling Analysis ===")
    scaling_df = evaluate_data_scaling(
        train_df, test_df,
        train_sizes=[5, 10, 20, 40, 80, 120, 160, 200, 240],
        k_values=[1, 2, 3, 4, 5],
        num_runs=10,
        target_column=args.target,
        numerical_features=num_feats,
        categorical_features=cat_feats,
    )
    scaling_df.to_csv(out_dir / "scaling_analysis.csv", index=False)
    print(f"  Saved {len(scaling_df)} rows to scaling_analysis.csv")

    # ---- 3. Missingness Evaluation ----------------------------------------
    print("\n=== Missingness Evaluation ===")
    miss_df = evaluate_missingness(
        train_df, test_df,
        mask_percentages=list(np.arange(0, 1.0, 0.1)),
        train_sizes=[5, 10, 20, 40, 80, 120, 160, 200, 240],
        k=5,
        num_seeds=10,
        target_column=args.target,
        numerical_features=num_feats,
        categorical_features=cat_feats,
    )
    miss_df.to_csv(out_dir / "missingness_analysis.csv", index=False)
    print(f"  Saved {len(miss_df)} rows to missingness_analysis.csv")

    # ---- 4. Runtime Efficiency --------------------------------------------
    print("\n=== Runtime Efficiency ===")
    runtime_df = evaluate_runtime(
        train_df, test_df,
        train_sizes=[10, 50, 100, 500, 1000, 2000, 5000],
        k=5,
        num_repeats=3,
        target_column=args.target,
        numerical_features=num_feats,
        categorical_features=cat_feats,
    )
    runtime_df.to_csv(out_dir / "runtime_analysis.csv", index=False)
    print(f"  Saved {len(runtime_df)} rows to runtime_analysis.csv")

    # Print summary
    summary = runtime_df.groupby("train_size")["per_query_ms"].mean()
    print("  Avg per-query time (ms):")
    for size, ms in summary.items():
        print(f"    N={size}: {ms:.2f} ms")

    # ---- 5. Baseline Comparison -------------------------------------------
    print("\n=== Baseline Comparison ===")
    baseline_df = evaluate_baselines(
        train_df, test_df,
        target_column=args.target,
        k_values=[1, 3, 5],
        numerical_features=num_feats,
        categorical_features=cat_feats,
    )
    baseline_df.to_csv(out_dir / "baselines.csv", index=False)
    print(baseline_df.to_string(index=False))

    # ---- 6. Cross-Validation ----------------------------------------------
    print("\n=== 5-Fold Cross-Validation ===")
    cv_df = cross_validate(
        df, n_folds=5, k=5,
        weight_strategy=WeightStrategy.COMPLETENESS,
        target_column=args.target,
        numerical_features=num_feats,
        categorical_features=cat_feats,
    )
    cv_df.to_csv(out_dir / "cross_validation.csv", index=False)
    print(cv_df.to_string(index=False))
    print(f"  Mean MAPE: {cv_df['mape'].mean() * 100:.2f}%")
    print(f"  Mean R²:   {cv_df['r2'].mean():.4f}")

    print(f"\nAll results saved to {out_dir.resolve()}/")


if __name__ == "__main__":
    main()
