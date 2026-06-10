"""
weighted_knn — Weighted k-Nearest Neighbor Estimator for Environmental Impact.

A k-NN framework that handles heterogeneous attribute availability,
mixed numerical/categorical features, and provides Gaussian uncertainty
estimates (point prediction + 95 % confidence interval).

Quick start::

    from estimation_using_textual_features import WeightedKNNEstimator, WeightStrategy
    from estimation_using_textual_features.data_loaders import load_electricity_data

    df = load_electricity_data("data/ElectricityMaps.txt", include_zone=False)
    train, test = train_test_split(df, test_size=0.2, random_state=42)

    model = WeightedKNNEstimator(k=5, weight_strategy=WeightStrategy.COMPLETENESS)
    model.fit(train)
    results = model.predict(test)
    print(results.summary())
"""

from .weighted_knn_estimator import (
    WeightedKNNEstimator,
    WeightStrategy,
    PredictionResult,
    PredictionResults,
    FeaturePreprocessor,
    apply_random_masking,
)

__all__ = [
    "WeightedKNNEstimator",
    "WeightStrategy",
    "PredictionResult",
    "PredictionResults",
    "FeaturePreprocessor",
    "apply_random_masking",
]
