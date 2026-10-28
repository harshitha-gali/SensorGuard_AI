
"""Train baseline and noise-aware sensor fault classifiers."""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from src.corruption import corrupt_sensor_data
from src.data_generation import FEATURES, TARGET, generate_sensor_data


RANDOM_STATE = 42
MODEL_DIR = Path("models")


def build_model() -> Pipeline:
    """Build a consistent preprocessing and classification pipeline."""
    return Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=150,
                    random_state=RANDOM_STATE,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def make_noise_augmented_data(
    X: pd.DataFrame,
    y: pd.Series,
    noise_level: float = 0.15,
    copies: int = 2,
) -> tuple[pd.DataFrame, pd.Series]:
    """Augment training features with simulated random sensor noise."""
    if copies < 1:
        raise ValueError("copies must be at least 1.")

    feature_frames = [X.reset_index(drop=True)]
    label_frames = [y.reset_index(drop=True)]

    for copy_number in range(copies):
        corrupted = corrupt_sensor_data(
            X,
            noise_level=noise_level,
            random_state=RANDOM_STATE + copy_number + 1,
        )

        feature_frames.append(corrupted[FEATURES].reset_index(drop=True))
        label_frames.append(y.reset_index(drop=True))

    return (
        pd.concat(feature_frames, ignore_index=True),
        pd.concat(label_frames, ignore_index=True),
    )


def train_models(
    n_samples: int = 4000,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
    save_models: bool = True,
) -> dict:
    """Train both models using one shared, stratified data split."""
    data = generate_sensor_data(
        n_samples=n_samples,
        random_state=random_state,
    )

    # Split before augmentation to prevent test-data leakage.
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        data[FEATURES],
        data[TARGET],
        test_size=test_size,
        random_state=random_state,
        stratify=data[TARGET],
    )

    baseline = build_model()
    baseline.fit(X_train, y_train)

    X_augmented, y_augmented = make_noise_augmented_data(
        X_train,
        y_train,
        noise_level=0.15,
        copies=2,
    )

    noise_aware = build_model()
    noise_aware.fit(X_augmented, y_augmented)

    if save_models:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(baseline, MODEL_DIR / "baseline_model.joblib")
        joblib.dump(noise_aware, MODEL_DIR / "noise_aware_model.joblib")

    return {
        "baseline": baseline,
        "noise_aware": noise_aware,
        "X_test": X_test,
        "y_test": y_test,
        "X_train": X_train,
        "y_train": y_train,
    }


if __name__ == "__main__":
    trained = train_models()

    print("Model training completed successfully.")
    print("Training samples:", len(trained["X_train"]))
    print("Test samples:", len(trained["X_test"]))
    print("Baseline model:", type(trained["baseline"].named_steps["classifier"]).__name__)
    print("Noise-aware model:", type(trained["noise_aware"].named_steps["classifier"]).__name__)
    print("Saved models to:", MODEL_DIR.resolve())
