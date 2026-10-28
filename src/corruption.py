
"""Simulate common sensor-data corruption scenarios."""

import numpy as np
import pandas as pd

from src.data_generation import FEATURES


def corrupt_sensor_data(
    data: pd.DataFrame,
    noise_level: float = 0.0,
    missing_rate: float = 0.0,
    drift_level: float = 0.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Return a corrupted copy of sensor data.

    noise_level: noise standard deviation in feature-scale units.
    missing_rate: fraction of individual readings replaced by NaN.
    drift_level: fraction of each feature's standard deviation added
                 as a systematic shift to its readings.
    """
    if not 0.0 <= missing_rate <= 1.0:
        raise ValueError("missing_rate must be between 0 and 1.")

    if noise_level < 0 or drift_level < 0:
        raise ValueError("Noise and drift levels must be non-negative.")

    corrupted = data.copy()
    rng = np.random.default_rng(random_state)

    for column in FEATURES:
        values = corrupted[column].to_numpy(dtype=float)
        scale = values.std(ddof=0)

        # Add random noise relative to each feature's own scale.
        if noise_level > 0:
            values += rng.normal(
                0.0, noise_level * scale, size=len(values)
            )

        # Apply a consistent shift to simulate sensor drift.
        if drift_level > 0:
            values += drift_level * scale

        corrupted[column] = values

    # Mask individual readings, leaving class labels unchanged.
    if missing_rate > 0:
        mask = rng.random(
            (len(corrupted), len(FEATURES))
        ) < missing_rate

        feature_values = corrupted[FEATURES].to_numpy(copy=True)
        feature_values[mask] = np.nan
        corrupted.loc[:, FEATURES] = feature_values

    return corrupted
