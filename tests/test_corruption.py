
"""Tests for simulated sensor corruption."""

import numpy as np
import pandas as pd
import pytest

from src.corruption import corrupt_sensor_data
from src.data_generation import FEATURES, TARGET, generate_sensor_data


def test_corruption_preserves_shape_and_labels():
    data = generate_sensor_data(n_samples=100)

    corrupted = corrupt_sensor_data(
        data,
        noise_level=0.2,
        random_state=42,
    )

    assert corrupted.shape == data.shape
    pd.testing.assert_series_equal(corrupted[TARGET], data[TARGET])


def test_noise_changes_sensor_readings():
    data = generate_sensor_data(n_samples=100)

    corrupted = corrupt_sensor_data(
        data,
        noise_level=0.2,
        random_state=42,
    )

    assert not np.allclose(
        data[FEATURES].to_numpy(),
        corrupted[FEATURES].to_numpy(),
    )


def test_missing_rate_inserts_missing_values():
    data = generate_sensor_data(n_samples=100)

    corrupted = corrupt_sensor_data(
        data,
        missing_rate=0.2,
        random_state=42,
    )

    missing_count = corrupted[FEATURES].isna().sum().sum()

    assert missing_count > 0
    assert missing_count < len(data) * len(FEATURES)


def test_zero_corruption_preserves_data():
    data = generate_sensor_data(n_samples=100)

    corrupted = corrupt_sensor_data(data)

    pd.testing.assert_frame_equal(corrupted, data)


def test_invalid_missing_rate_raises_error():
    data = generate_sensor_data(n_samples=100)

    with pytest.raises(ValueError):
        corrupt_sensor_data(data, missing_rate=1.5)


def test_negative_noise_level_raises_error():
    data = generate_sensor_data(n_samples=100)

    with pytest.raises(ValueError):
        corrupt_sensor_data(data, noise_level=-0.1)
