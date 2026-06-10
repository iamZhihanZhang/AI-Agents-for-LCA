"""
Evaluation utilities for the Weighted kNN Estimator.

Provides functions for:
    - Data scaling analysis (varying training set size)
    - Synthetic missingness evaluation
    - Runtime efficiency benchmarking
    - Baseline comparisons (sklearn SVM and kNN)
    - Cross-validation

All functions return ``pandas.DataFrame`` results suitable for plotting
and CSV export.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)

from .weighted_knn_estimator import (
    PredictionResults,
    WeightedKNNEstimator,
    WeightStrategy,
    apply_random_masking,
)


# ---------------------------------------------------------------------------
# 1. Data Scaling Analysis
# ---------------------------------------------------------------------------

def evaluate_data_scaling(
    full_train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_sizes: list[int] | None = None,
    k_values: list[int] | None = None,
    num_runs: int = 10,
    target_column: str = "carbon_intensity",
    weight_strategy: WeightStrategy = WeightStrategy.COMPLETENESS,
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    """Evaluate prediction performance across varying training set sizes.

    For each combination of ``train_size`` and ``k``, the training set is
    randomly sub-sampled ``num_runs`` times with different seeds.  MAPE
    and standard deviation of per-sample absolute percentage errors are
    recorded.

    Parameters:
        full_train_df: Full training DataFrame.
        test_df: Held-out test DataFrame (fixed across all runs).
        train_sizes: List of training set sizes to evaluate.
            Default: ``[5, 10, 20, 40, 80, 120, 160, 200, 240]``.
        k_values: List of k (neighbor count) values to try.
            Default: ``[1, 2, 3, 4, 5]``.
        num_runs: Number of random sub-sampling repetitions per setting.
        target_column: Name of the target column.
        weight_strategy: Weighting strategy to use.
        numerical_features: Explicit list of numerical feature columns.
        categorical_features: Explicit list of categorical feature columns.

    Returns:
        DataFrame with columns ``[k, train_size, seed, mape, std_ape]``.
    """
    if train_sizes is None:
        train_sizes = [5, 10, 20, 40, 80, 120, 160, 200, 240]
    if k_values is None:
        k_values = [1, 2, 3, 4, 5]

    records: list[dict] = []

    for k in k_values:
        for size in train_sizes:
            if size > len(full_train_df):
                continue
            for seed in range(num_runs):
                sampled = full_train_df.sample(
                    n=size, random_state=seed
                ).reset_index(drop=True)

                model = WeightedKNNEstimator(
                    k=k,
                    weight_strategy=weight_strategy,
                    target_column=target_column,
                    numerical_features=numerical_features,
                    categorical_features=categorical_features,
                )
                model.fit(sampled)
                results = model.predict(test_df)

                # Per-sample absolute percentage errors
                apes = np.abs(
                    (np.array(results.true_values) - np.array(results.predicted_values))
                    / np.array(results.true_values)
                )

                records.append(
                    {
                        "k": k,
                        "train_size": size,
                        "seed": seed,
                        "mape": results.mape,
                        "std_ape": float(np.std(apes)),
                    }
                )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 2. Synthetic Missingness Evaluation
# ---------------------------------------------------------------------------

def evaluate_missingness(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    mask_percentages: list[float] | None = None,
    train_sizes: list[int] | None = None,
    weight_strategies: list[WeightStrategy] | None = None,
    k: int = 5,
    num_seeds: int = 10,
    target_column: str = "carbon_intensity",
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    mask_train: bool = True,
    mask_test: bool = True,
) -> pd.DataFrame:
    """Evaluate robustness to synthetic missingness across configurations.

    For each combination of ``train_size``, ``mask_percentage``, and
    ``weight_strategy``, features are randomly masked at the specified
    rate for ``num_seeds`` repetitions.

    Parameters:
        train_df: Full training DataFrame.
        test_df: Held-out test DataFrame.
        mask_percentages: Fractions of features to mask per row.
            Default: ``np.arange(0, 1, 0.1)``.
        train_sizes: Training set sizes to evaluate.
            Default: ``[5, 10, 20, 40, 80, 120, 160, 200, 240]``.
        weight_strategies: List of ``WeightStrategy`` values to compare.
            Default: all strategies.
        k: Number of nearest neighbors.
        num_seeds: Number of random seeds per configuration.
        target_column: Name of the target column.
        numerical_features: Explicit numerical feature column list.
        categorical_features: Explicit categorical feature column list.
        mask_train: Whether to apply masking to the training set.
        mask_test: Whether to apply masking to the test set.

    Returns:
        DataFrame with columns ``[train_size, mask_pct, strategy, seed, mape]``.
    """
    if mask_percentages is None:
        mask_percentages = list(np.arange(0, 1.0, 0.1))
    if train_sizes is None:
        train_sizes = [5, 10, 20, 40, 80, 120, 160, 200, 240]
    if weight_strategies is None:
        weight_strategies = list(WeightStrategy)

    # Identify feature columns (everything except target)
    feature_cols = [c for c in train_df.columns if c != target_column]

    records: list[dict] = []

    for size in train_sizes:
        if size > len(train_df):
            continue
        for mask_pct in mask_percentages:
            for strategy in weight_strategies:
                for seed in range(num_seeds):
                    # Sub-sample training set
                    sampled_train = train_df.sample(
                        n=size, random_state=seed
                    ).reset_index(drop=True)

                    # Apply masking
                    if mask_train and mask_pct > 0:
                        sampled_train = apply_random_masking(
                            sampled_train, feature_cols, mask_pct,
                            random_state=seed,
                        )
                    masked_test = test_df.copy()
                    if mask_test and mask_pct > 0:
                        masked_test = apply_random_masking(
                            masked_test, feature_cols, mask_pct,
                            random_state=seed + 1000,
                        )

                    model = WeightedKNNEstimator(
                        k=k,
                        weight_strategy=strategy,
                        target_column=target_column,
                        numerical_features=numerical_features,
                        categorical_features=categorical_features,
                    )
                    model.fit(sampled_train)
                    results = model.predict(masked_test)

                    records.append(
                        {
                            "train_size": size,
                            "mask_pct": round(mask_pct, 2),
                            "strategy": strategy.value,
                            "seed": seed,
                            "mape": results.mape,
                        }
                    )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 3. Runtime Efficiency Benchmarking
# ---------------------------------------------------------------------------

def evaluate_runtime(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    train_sizes: list[int] | None = None,
    k: int = 5,
    num_repeats: int = 3,
    target_column: str = "carbon_intensity",
    weight_strategy: WeightStrategy = WeightStrategy.COMPLETENESS,
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    """Benchmark per-query inference time as a function of training set size.

    For training sizes exceeding the available data, the training set is
    replicated (concatenated copies) to simulate larger databases.  This
    preserves the computational cost of distance calculations and sorting
    without adding new information.

    Parameters:
        train_df: Training DataFrame.
        test_df: Test DataFrame (a small subset is used for timing).
        train_sizes: List of (possibly synthetic) training sizes.
            Default: ``[10, 50, 100, 500, 1000, 2000, 5000]``.
        k: Number of nearest neighbors.
        num_repeats: Number of timing repetitions per size for averaging.
        target_column: Name of the target column.
        weight_strategy: Weighting strategy.
        numerical_features: Numerical feature column list.
        categorical_features: Categorical feature column list.

    Returns:
        DataFrame with columns ``[train_size, repeat, total_seconds,
        per_query_ms, n_test_queries]``.
    """
    if train_sizes is None:
        train_sizes = [10, 50, 100, 500, 1000, 2000, 5000]

    # Use at most 20 test queries for timing (keep it fast)
    timing_test = test_df.head(min(20, len(test_df))).copy()
    n_full = len(train_df)
    records: list[dict] = []

    for size in train_sizes:
        # Build a training set of the desired size via replication
        if size <= n_full:
            synthetic_train = train_df.sample(
                n=size, random_state=42
            ).reset_index(drop=True)
        else:
            copies_needed = int(np.ceil(size / n_full))
            synthetic_train = pd.concat(
                [train_df] * copies_needed, ignore_index=True
            ).iloc[:size]

        for rep in range(num_repeats):
            model = WeightedKNNEstimator(
                k=k,
                weight_strategy=weight_strategy,
                target_column=target_column,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
            )
            model.fit(synthetic_train)

            start = time.perf_counter()
            model.predict(timing_test)
            elapsed = time.perf_counter() - start

            n_queries = len(timing_test)
            records.append(
                {
                    "train_size": size,
                    "repeat": rep,
                    "total_seconds": elapsed,
                    "per_query_ms": (elapsed / n_queries) * 1000,
                    "n_test_queries": n_queries,
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Baseline Comparisons (sklearn SVM and kNN)
# ---------------------------------------------------------------------------

def evaluate_baselines(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_column: str = "carbon_intensity",
    k_values: list[int] | None = None,
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
) -> pd.DataFrame:
    """Compare sklearn SVM and kNN baselines against the weighted kNN.

    All three model families are trained on the same data.  For sklearn
    models, NaN values are imputed with column means and categorical
    features are one-hot encoded.

    Parameters:
        train_df: Training DataFrame.
        test_df: Test DataFrame (must include the target column).
        target_column: Name of the target column.
        k_values: List of k values for kNN models.
            Default: ``[1, 3, 5]``.
        numerical_features: Numerical feature column list.
        categorical_features: Categorical feature column list.

    Returns:
        DataFrame with columns ``[model, k, mape, mae, mse, r2]``.
    """
    if k_values is None:
        k_values = [1, 3, 5]

    # -- prepare features for sklearn baselines --
    feature_cols = [c for c in train_df.columns if c != target_column]
    if numerical_features is None:
        numerical_features = [
            c for c in feature_cols
            if pd.api.types.is_numeric_dtype(train_df[c])
        ]
    if categorical_features is None:
        categorical_features = [
            c for c in feature_cols
            if not pd.api.types.is_numeric_dtype(train_df[c])
        ]

    def _prepare_sklearn(df: pd.DataFrame):
        """One-hot encode categoricals, impute NaN, scale numerics."""
        parts = []
        num_cols = [c for c in numerical_features if c in df.columns]
        cat_cols = [c for c in categorical_features if c in df.columns]
        if num_cols:
            parts.append(df[num_cols].copy())
        for col in cat_cols:
            dummies = pd.get_dummies(df[col], prefix=col, dummy_na=False)
            parts.append(dummies)
        if not parts:
            raise ValueError("No feature columns found.")
        combined = pd.concat(parts, axis=1).astype(float)
        return combined

    X_train_raw = _prepare_sklearn(train_df)
    X_test_raw = _prepare_sklearn(test_df)

    # Align columns (test may have unseen categories)
    X_train_raw, X_test_raw = X_train_raw.align(
        X_test_raw, join="left", axis=1, fill_value=0
    )

    # Impute NaN with column mean, then scale
    train_means = X_train_raw.mean()
    X_train_filled = X_train_raw.fillna(train_means)
    X_test_filled = X_test_raw.fillna(train_means)

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train_filled)
    X_test_sc = scaler.transform(X_test_filled)

    y_train = train_df[target_column].values
    y_test = test_df[target_column].values

    records: list[dict] = []

    # -- sklearn kNN baselines --
    for k in k_values:
        skl_knn = KNeighborsRegressor(n_neighbors=k, weights="distance")
        skl_knn.fit(X_train_sc, y_train)
        y_pred = skl_knn.predict(X_test_sc)
        records.append(
            {
                "model": "sklearn_kNN",
                "k": k,
                "mape": mean_absolute_percentage_error(y_test, y_pred),
                "mae": mean_absolute_error(y_test, y_pred),
                "mse": mean_squared_error(y_test, y_pred),
                "r2": r2_score(y_test, y_pred),
            }
        )

    # -- SVM baseline (RBF kernel) --
    svm = SVR(kernel="rbf", C=1.0, epsilon=0.1)
    svm.fit(X_train_sc, y_train)
    y_pred_svm = svm.predict(X_test_sc)
    records.append(
        {
            "model": "sklearn_SVM",
            "k": None,
            "mape": mean_absolute_percentage_error(y_test, y_pred_svm),
            "mae": mean_absolute_error(y_test, y_pred_svm),
            "mse": mean_squared_error(y_test, y_pred_svm),
            "r2": r2_score(y_test, y_pred_svm),
        }
    )

    # -- Weighted kNN (ours) --
    for k in k_values:
        for strategy in WeightStrategy:
            model = WeightedKNNEstimator(
                k=k,
                weight_strategy=strategy,
                target_column=target_column,
                numerical_features=numerical_features,
                categorical_features=categorical_features,
            )
            model.fit(train_df)
            results = model.predict(test_df)
            records.append(
                {
                    "model": f"weighted_kNN_{strategy.value}",
                    "k": k,
                    "mape": results.mape,
                    "mae": results.mae,
                    "mse": results.mse,
                    "r2": results.r2,
                }
            )

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 5. Cross-Validation
# ---------------------------------------------------------------------------

def cross_validate(
    df: pd.DataFrame,
    n_folds: int = 5,
    k: int = 5,
    weight_strategy: WeightStrategy = WeightStrategy.COMPLETENESS,
    target_column: str = "carbon_intensity",
    numerical_features: list[str] | None = None,
    categorical_features: list[str] | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run k-fold cross-validation and return per-fold metrics.

    Parameters:
        df: Full dataset (features + target).
        n_folds: Number of CV folds.
        k: Number of nearest neighbors.
        weight_strategy: Neighbor weighting strategy.
        target_column: Name of the target column.
        numerical_features: Numerical feature column list.
        categorical_features: Categorical feature column list.
        random_state: Seed for fold splitting.

    Returns:
        DataFrame with columns ``[fold, mape, mae, mse, r2]``.
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    records: list[dict] = []

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(df)):
        train_fold = df.iloc[train_idx].reset_index(drop=True)
        test_fold = df.iloc[test_idx].reset_index(drop=True)

        model = WeightedKNNEstimator(
            k=k,
            weight_strategy=weight_strategy,
            target_column=target_column,
            numerical_features=numerical_features,
            categorical_features=categorical_features,
        )
        model.fit(train_fold)
        results = model.predict(test_fold)

        records.append(
            {
                "fold": fold_idx + 1,
                "mape": results.mape,
                "mae": results.mae,
                "mse": results.mse,
                "r2": results.r2,
            }
        )

    return pd.DataFrame(records)
