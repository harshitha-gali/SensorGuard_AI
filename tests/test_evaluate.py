
"""Tests for model evaluation."""

from src.evaluate import evaluate_models


def test_evaluation_returns_all_scenarios_and_models():
    results = evaluate_models(save_outputs=False)

    assert len(results) == 10
    assert set(results["scenario"]) == {
        "Clean",
        "Noise",
        "Missing Values",
        "Sensor Drift",
        "Combined",
    }
    assert set(results["model"]) == {
        "Baseline",
        "Noise-Aware",
    }


def test_evaluation_metrics_are_within_valid_ranges():
    results = evaluate_models(save_outputs=False)

    assert results["accuracy"].between(0, 1).all()
    assert results["macro_f1"].between(0, 1).all()


def test_clean_accuracy_drop_is_zero():
    results = evaluate_models(save_outputs=False)

    clean_results = results[results["scenario"] == "Clean"]

    assert len(clean_results) == 2
    assert (clean_results["accuracy_drop_percentage_points"] == 0).all()
