
"""Tests for synthetic sensor data generation."""

import pandas as pd
import pytest

from src.data_generation import (
    CLASS_NAMES,
    FEATURES,
    TARGET,
    generate_sensor_data,
)


def test_generated_data_has_expected_shape():
    data = generate_sensor_data(n_samples=100)

    assert data.shape == (100, 5)


def test_generated_data_has_expected_columns():
    data = generate_sensor_data()

    assert list(data.columns) == FEATURES + [TARGET]


def test_generated_data_contains_all_classes():
    data = generate_sensor_data()

    assert set(data[TARGET].unique()) == set(CLASS_NAMES)


def test_generated_classes_are_balanced():
    data = generate_sensor_data()

    counts = data[TARGET].value_counts()

    assert counts.nunique() == 1


def test_generated_data_is_reproducible():
    first = generate_sensor_data(n_samples=100, random_state=42)
    second = generate_sensor_data(n_samples=100, random_state=42)

    pd.testing.assert_frame_equal(first, second)


def test_invalid_sample_count_raises_error():
    with pytest.raises(ValueError):
        generate_sensor_data(n_samples=3)
