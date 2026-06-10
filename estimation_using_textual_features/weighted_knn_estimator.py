"""
Weighted k-Nearest Neighbor Estimator for Environmental Impact Prediction.

This module implements a weighted k-NN framework that handles heterogeneous
attribute availability, mixed data types (numerical + categorical), and
provides Gaussian uncertainty estimates. It is designed for Life Cycle
Assessment (LCA) carbon footprint estimation but generalizes to any
regression task with incomplete, mixed-type features.

Reference:
    See the accompanying paper's Supplementary "Weighted k-Nearest Neighbor Estimator" section for the mathematical formulation.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.metrics.pairwise import nan_euclidean_distances
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Weighting strategies
# ---------------------------------------------------------------------------

class WeightStrategy(Enum):
    """Neighbor weighting strategies for the kNN estimator.

    Attributes:
        COMPLETENESS: Weight by total number of observed attributes per neighbor.
            Favors neighbors with richer documentation.
        SHARED_FEATURES: Weight by the number of attributes co-observed between
            the query and the neighbor. Useful when the query itself is sparse.
        HYBRID_COMPLETENESS: Combines inverse-distance weighting with
            completeness weighting (product of the two).
        HYBRID_SHARED: Combines inverse-distance weighting with shared-feature
            weighting (product of the two).
        UNIFORM: Equal weight for all k neighbors (standard kNN).
        INVERSE_DISTANCE: Weight inversely proportional to distance (classic
            distance-weighted kNN).
    """

    COMPLETENESS = "completeness"
    SHARED_FEATURES = "shared_features"
    HYBRID_COMPLETENESS = "hybrid_completeness"
    HYBRID_SHARED = "hybrid_shared"
    UNIFORM = "uniform"
    INVERSE_DISTANCE = "inverse_distance"


# ---------------------------------------------------------------------------
# Prediction result container
# ---------------------------------------------------------------------------

@dataclass
class PredictionResult:
    """Container for a single prediction."""

    true_value: float
    predicted_value: float
    ci_lower: float
    ci_upper: float


@dataclass
class PredictionResults:
    """Aggregated prediction results with evaluation metrics.

    Attributes:
        results: List of individual ``PredictionResult`` objects.
        mse: Mean Squared Error over all predictions.
        mae: Mean Absolute Error over all predictions.
        mape: Mean Absolute Percentage Error (as a fraction, e.g. 0.05 = 5 %).
        r2: R-squared score.
        elapsed_seconds: Total wall-clock time for all predictions.
    """

    results: list[PredictionResult] = field(default_factory=list)
    mse: float = 0.0
    mae: float = 0.0
    mape: float = 0.0
    r2: float = 0.0
    elapsed_seconds: float = 0.0

    # -- convenience properties --

    @property
    def true_values(self) -> list[float]:
        return [r.true_value for r in self.results]

    @property
    def predicted_values(self) -> list[float]:
        return [r.predicted_value for r in self.results]

    @property
    def confidence_intervals(self) -> list[tuple[float, float]]:
        return [(r.ci_lower, r.ci_upper) for r in self.results]

    def summary(self) -> dict:
        """Return a summary dictionary of evaluation metrics."""
        return {
            "mse": self.mse,
            "mae": self.mae,
            "mape": self.mape,
            "r2": self.r2,
            "elapsed_seconds": self.elapsed_seconds,
            "n_samples": len(self.results),
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert results to a pandas DataFrame."""
        rows = []
        for r in self.results:
            pct_err = (
                abs(r.true_value - r.predicted_value) / abs(r.true_value) * 100
                if r.true_value != 0
                else float("inf")
            )
            rows.append(
                {
                    "true_value": r.true_value,
                    "predicted_value": r.predicted_value,
                    "pct_error": pct_err,
                    "ci_lower": r.ci_lower,
                    "ci_upper": r.ci_upper,
                }
            )
        return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Feature preprocessor (handles mixed types)
# ---------------------------------------------------------------------------

class FeaturePreprocessor:
    """Preprocesses mixed numerical/categorical features.

    Numerical features are z-score normalized (fit only on observed values).
    Categorical features are one-hot encoded.  Missing values (NaN) are
    preserved through all transformations so that ``nan_euclidean_distances``
    can handle them downstream.

    Parameters:
        numerical_features: Column names of numerical attributes.
        categorical_features: Column names of categorical attributes.
    """

    def __init__(
        self,
        numerical_features: list[str],
        categorical_features: list[str] | None = None,
    ):
        self.numerical_features = list(numerical_features)
        self.categorical_features = list(categorical_features or [])
        self._scaler: StandardScaler | None = None
        self._cat_categories: dict[str, list] = {}
        self._feature_names_out: list[str] = []

    def fit(self, df: pd.DataFrame) -> "FeaturePreprocessor":
        """Learn scaling parameters and category vocabularies from *df*.

        Only observed (non-NaN) values are used.
        """
        # Numerical: fit StandardScaler ignoring NaN
        num_cols = [c for c in self.numerical_features if c in df.columns]
        if num_cols:
            self._scaler = StandardScaler()
            # StandardScaler doesn't natively handle NaN; compute manually
            self._num_means = df[num_cols].mean()
            self._num_stds = df[num_cols].std().replace(0, 1)  # avoid /0
        self._num_cols_fitted = num_cols

        # Categorical: learn all categories (including from NaN rows)
        cat_cols = [c for c in self.categorical_features if c in df.columns]
        for col in cat_cols:
            self._cat_categories[col] = sorted(
                df[col].dropna().unique().tolist()
            )
        self._cat_cols_fitted = cat_cols

        # Build output feature names
        self._feature_names_out = list(num_cols)
        for col in cat_cols:
            for cat in self._cat_categories[col]:
                self._feature_names_out.append(f"{col}__{cat}")

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform *df* into the unified feature space.

        Returns a DataFrame of shape ``(len(df), d_max)`` with NaN
        preserved for missing attributes.
        """
        parts: list[pd.DataFrame] = []

        # Numerical columns: z-score normalise, preserving NaN
        if self._num_cols_fitted:
            num_data = df[self._num_cols_fitted].copy().astype(float)
            num_data = (num_data - self._num_means) / self._num_stds
            parts.append(num_data.reset_index(drop=True))

        # Categorical columns: one-hot encode, preserving NaN
        for col in self._cat_cols_fitted:
            cats = self._cat_categories[col]
            ohe = pd.DataFrame(
                np.nan, index=range(len(df)), columns=[f"{col}__{c}" for c in cats]
            )
            for i, val in enumerate(df[col].values):
                if pd.isna(val):
                    # All one-hot columns stay NaN (attribute missing)
                    continue
                for c in cats:
                    ohe.iloc[i, cats.index(c)] = 1.0 if val == c else 0.0
            parts.append(ohe)

        result = pd.concat(parts, axis=1)
        result.columns = self._feature_names_out
        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @property
    def feature_names(self) -> list[str]:
        return list(self._feature_names_out)

    @property
    def d_max(self) -> int:
        return len(self._feature_names_out)


# ---------------------------------------------------------------------------
# Core estimator
# ---------------------------------------------------------------------------

class WeightedKNNEstimator:
    """Weighted k-Nearest Neighbor estimator with Gaussian uncertainty.

    This estimator:
    1. Preprocesses mixed numerical/categorical features via
       ``FeaturePreprocessor``.
    2. Computes NaN-aware Euclidean distances (``nan_euclidean_distances``
       from scikit-learn) so that products with different attribute subsets
       can still be compared.
    3. Selects the *k* nearest neighbors and assigns importance weights
       according to the chosen ``WeightStrategy``.
    4. Returns a Gaussian point estimate (weighted mean) and 95 %
       confidence interval (weighted standard deviation).

    Parameters:
        k: Number of nearest neighbors (default 5).
        weight_strategy: One of the ``WeightStrategy`` enum values.
        confidence_level: Confidence level for the interval (default 0.95).
        target_column: Name of the target variable in the training data.
        numerical_features: List of numerical attribute column names.
            If *None*, inferred at fit time as all float/int columns
            except the target.
        categorical_features: List of categorical attribute column names.
            If *None*, inferred at fit time as all object/category columns.
    """

    def __init__(
        self,
        k: int = 5,
        weight_strategy: WeightStrategy = WeightStrategy.COMPLETENESS,
        confidence_level: float = 0.95,
        target_column: str = "carbon_intensity",
        numerical_features: list[str] | None = None,
        categorical_features: list[str] | None = None,
    ):
        self.k = k
        self.weight_strategy = weight_strategy
        self.confidence_level = confidence_level
        self.target_column = target_column
        self._numerical_features = numerical_features
        self._categorical_features = categorical_features
        self._preprocessor: FeaturePreprocessor | None = None
        self._train_features: np.ndarray | None = None
        self._train_targets: np.ndarray | None = None
        self._train_masks: np.ndarray | None = None  # (N, d_max) bool
        self._raw_train_df: pd.DataFrame | None = None

    # ---- public API -------------------------------------------------------

    def fit(self, train_df: pd.DataFrame) -> "WeightedKNNEstimator":
        """Fit the estimator on training data.

        Parameters:
            train_df: DataFrame containing both feature columns and the
                target column.  May contain NaN in feature columns.

        Returns:
            self
        """
        num_feats, cat_feats = self._resolve_feature_lists(train_df)
        self._preprocessor = FeaturePreprocessor(num_feats, cat_feats)
        feature_df = self._preprocessor.fit_transform(
            train_df.drop(columns=[self.target_column])
        )
        self._train_features = feature_df.values.astype(float)
        self._train_targets = train_df[self.target_column].values.astype(float)
        self._train_masks = ~np.isnan(self._train_features)  # True = observed
        self._raw_train_df = train_df.copy()
        return self

    def predict_single(
        self, query: pd.Series | pd.DataFrame
    ) -> tuple[float, float, float]:
        """Predict target value and 95 % CI for a single query.

        Parameters:
            query: A single row of feature values (may contain NaN).

        Returns:
            (mean_estimate, ci_lower, ci_upper)
        """
        if isinstance(query, pd.Series):
            query = query.to_frame().T

        # Drop target if accidentally included
        if self.target_column in query.columns:
            query = query.drop(columns=[self.target_column])

        q_transformed = self._preprocessor.transform(query).values.astype(float)
        q_mask = ~np.isnan(q_transformed)  # (1, d_max)

        # Compute NaN-aware Euclidean distances
        distances = nan_euclidean_distances(
            self._train_features, q_transformed
        ).ravel()  # (N,)

        # Select k nearest neighbors
        k = min(self.k, len(distances))
        neighbor_idx = np.argsort(distances)[:k]
        neighbor_targets = self._train_targets[neighbor_idx]
        neighbor_distances = distances[neighbor_idx]

        # Compute weights
        weights = self._compute_weights(
            neighbor_idx, neighbor_distances, q_mask
        )

        # Normalise weights
        w_sum = weights.sum()
        if w_sum == 0:
            weights = np.ones(k) / k
        else:
            weights = weights / w_sum

        # Weighted Gaussian estimate
        mu = np.average(neighbor_targets, weights=weights)
        variance = np.average((neighbor_targets - mu) ** 2, weights=weights)

        if variance == 0:
            return mu, mu, mu

        sigma = np.sqrt(variance)
        ci_lower, ci_upper = stats.norm.interval(
            self.confidence_level, loc=mu, scale=sigma
        )
        return float(mu), float(ci_lower), float(ci_upper)

    def predict(self, test_df: pd.DataFrame) -> PredictionResults:
        """Predict target values for all rows in *test_df*.

        Parameters:
            test_df: DataFrame with the same feature columns as training
                data, plus (optionally) the target column for evaluation.

        Returns:
            ``PredictionResults`` containing per-sample predictions,
            confidence intervals, and aggregate metrics.
        """
        has_target = self.target_column in test_df.columns

        results_list: list[PredictionResult] = []
        start = time.perf_counter()

        for idx in range(len(test_df)):
            row = test_df.iloc[[idx]]
            true_val = (
                float(row[self.target_column].iloc[0]) if has_target else np.nan
            )
            mu, ci_lo, ci_hi = self.predict_single(row)
            results_list.append(
                PredictionResult(true_val, mu, ci_lo, ci_hi)
            )

        elapsed = time.perf_counter() - start

        # Compute aggregate metrics (only if ground truth is available)
        pr = PredictionResults(results=results_list, elapsed_seconds=elapsed)
        if has_target:
            y_true = pr.true_values
            y_pred = pr.predicted_values
            pr.mse = mean_squared_error(y_true, y_pred)
            pr.mae = mean_absolute_error(y_true, y_pred)
            pr.mape = mean_absolute_percentage_error(y_true, y_pred)
            pr.r2 = r2_score(y_true, y_pred)

        return pr

    # ---- private helpers ---------------------------------------------------

    def _resolve_feature_lists(
        self, df: pd.DataFrame
    ) -> tuple[list[str], list[str]]:
        """Auto-detect numerical and categorical columns if not specified."""
        cols = [c for c in df.columns if c != self.target_column]

        if self._numerical_features is not None:
            num = [c for c in self._numerical_features if c in cols]
        else:
            num = [
                c
                for c in cols
                if pd.api.types.is_numeric_dtype(df[c])
            ]

        if self._categorical_features is not None:
            cat = [c for c in self._categorical_features if c in cols]
        else:
            cat = [
                c
                for c in cols
                if not pd.api.types.is_numeric_dtype(df[c])
            ]

        return num, cat

    def _compute_weights(
        self,
        neighbor_idx: np.ndarray,
        neighbor_distances: np.ndarray,
        query_mask: np.ndarray,
    ) -> np.ndarray:
        """Compute neighbor weights according to the chosen strategy.

        Parameters:
            neighbor_idx: Indices into the training set for the k neighbors.
            neighbor_distances: Distances of the k neighbors to the query.
            query_mask: Boolean mask ``(1, d_max)`` indicating which features
                are observed in the query.

        Returns:
            1-D array of unnormalised weights for each neighbor.
        """
        k = len(neighbor_idx)
        masks = self._train_masks[neighbor_idx]  # (k, d_max)
        q_mask = query_mask.ravel()  # (d_max,)

        strategy = self.weight_strategy

        if strategy == WeightStrategy.UNIFORM:
            return np.ones(k)

        if strategy == WeightStrategy.INVERSE_DISTANCE:
            inv_d = np.where(
                neighbor_distances > 0, 1.0 / neighbor_distances, 1e12
            )
            return inv_d

        # Completeness: number of observed attributes per neighbor
        completeness = masks.sum(axis=1).astype(float)  # (k,)

        if strategy == WeightStrategy.COMPLETENESS:
            return completeness

        # Shared features: co-observed attributes between query and neighbor
        shared = (masks & q_mask).sum(axis=1).astype(float)  # (k,)

        if strategy == WeightStrategy.SHARED_FEATURES:
            return shared

        inv_d = np.where(
            neighbor_distances > 0, 1.0 / neighbor_distances, 1e12
        )

        if strategy == WeightStrategy.HYBRID_COMPLETENESS:
            return inv_d * completeness

        if strategy == WeightStrategy.HYBRID_SHARED:
            return inv_d * shared

        raise ValueError(f"Unknown weight strategy: {strategy}")


# ---------------------------------------------------------------------------
# Synthetic missingness utility
# ---------------------------------------------------------------------------

def apply_random_masking(
    df: pd.DataFrame,
    feature_columns: list[str],
    mask_fraction: float,
    random_state: int = 42,
) -> pd.DataFrame:
    """Randomly mask a fraction of feature values to NaN.

    For each row, ``floor(mask_fraction * len(feature_columns))`` randomly
    selected feature columns are set to NaN.  The target column and any
    non-feature columns are left untouched.

    Parameters:
        df: Input DataFrame.
        feature_columns: Columns eligible for masking.
        mask_fraction: Fraction of features to mask per row (0 to 1).
        random_state: Base random seed (combined with row index for
            per-row variation).

    Returns:
        A copy of *df* with masked entries.
    """
    masked = df.copy()
    cols = [c for c in feature_columns if c in masked.columns]
    n_to_mask = int(len(cols) * mask_fraction)
    if n_to_mask == 0:
        return masked

    rng = np.random.RandomState(random_state)
    for i in range(len(masked)):
        chosen = rng.choice(len(cols), size=n_to_mask, replace=False)
        for j in chosen:
            masked.iat[i, masked.columns.get_loc(cols[j])] = np.nan

    return masked
