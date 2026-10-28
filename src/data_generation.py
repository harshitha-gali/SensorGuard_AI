
"""Generate reproducible synthetic industrial sensor data."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = [
    "vibration",
    "temperature",
    "pressure",
    "rotational_speed",
]

TARGET = "fault_type"

CLASS_NAMES = [
    "Normal",
    "Overheating",
    "Excessive_Vibration",
    "Pressure_Anomaly",
]


def generate_sensor_data(
    n_samples: int = 4000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create synthetic sensor readings with four operating conditions."""
    if n_samples < len(CLASS_NAMES):
        raise ValueError("n_samples must be at least 4.")

    rng = np.random.default_rng(random_state)

    # Balance the four classes as evenly as possible.
    labels = np.resize(np.array(CLASS_NAMES), n_samples)
    rng.shuffle(labels)

    # Baseline readings represent normal machine operation.
    vibration = rng.normal(2.0, 0.35, n_samples)
    temperature = rng.normal(60.0, 4.0, n_samples)
    pressure = rng.normal(100.0, 5.0, n_samples)
    rotational_speed = rng.normal(1500.0, 80.0, n_samples)

    # Introduce a characteristic change for each simulated fault.
    overheating = labels == "Overheating"
    vibration_fault = labels == "Excessive_Vibration"
    pressure_fault = labels == "Pressure_Anomaly"

    temperature[overheating] += rng.normal(
        25.0, 4.0, overheating.sum()
    )
    vibration[vibration_fault] += rng.normal(
        3.0, 0.5, vibration_fault.sum()
    )
    pressure[pressure_fault] += rng.choice(
        [-1, 1], pressure_fault.sum()
    ) * rng.normal(25.0, 4.0, pressure_fault.sum())

    # Small operating variation prevents perfectly fixed readings.
    rotational_speed += rng.normal(0.0, 15.0, n_samples)

    return pd.DataFrame(
        {
            "vibration": np.clip(vibration, 0.1, None),
            "temperature": temperature,
            "pressure": pressure,
            "rotational_speed": np.clip(
                rotational_speed, 0.0, None
            ),
            TARGET: labels,
        }
    )


def split_sensor_data(
    data: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Split features and labels into stratified train/test sets."""
    if TARGET not in data.columns:
        raise ValueError(f"Dataset must contain '{TARGET}'.")

    if data[TARGET].nunique() < 2:
        raise ValueError("At least two classes are required.")

    X = data[FEATURES]
    y = data[TARGET]

    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
