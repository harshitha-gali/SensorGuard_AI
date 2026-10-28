
"""Tests for sensor fault model training."""

from src.train import train_models


def test_training_returns_both_models():
    result = train_models(
        n_samples=200,
        save_models=False,
    )

    assert "baseline" in result
    assert "noise_aware" in result


def test_training_returns_expected_data_split():
    result = train_models(
        n_samples=200,
        test_size=0.2,
        save_models=False,
    )

    assert len(result["X_train"]) == 160
    assert len(result["X_test"]) == 40
    assert len(result["y_train"]) == 160
    assert len(result["y_test"]) == 40


def test_training_models_can_make_predictions():
    result = train_models(
        n_samples=200,
        save_models=False,
    )

    baseline_predictions = result["baseline"].predict(
        result["X_test"]
    )
    noise_aware_predictions = result["noise_aware"].predict(
        result["X_test"]
    )

    assert len(baseline_predictions) == len(result["y_test"])
    assert len(noise_aware_predictions) == len(result["y_test"])
