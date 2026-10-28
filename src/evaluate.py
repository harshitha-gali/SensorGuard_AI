
"""Evaluate sensor fault classifiers under simulated corruption."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    ConfusionMatrixDisplay,
)

from src.corruption import corrupt_sensor_data
from src.data_generation import FEATURES
from src.train import train_models


RESULTS_DIR = Path("results")
FIGURES_DIR = RESULTS_DIR / "figures"

SCENARIOS = {
    "Clean": {},
    "Noise": {"noise_level": 0.30},
    "Missing Values": {"missing_rate": 0.15},
    "Sensor Drift": {"drift_level": 0.75},
    "Combined": {
        "noise_level": 0.30,
        "missing_rate": 0.15,
        "drift_level": 0.75,
    },
}


def evaluate_custom_corruption(
    trained: dict,
    noise_level: float = 0.0,
    missing_rate: float = 0.0,
    drift_level: float = 0.0,
    random_state: int = 42,
) -> pd.DataFrame:
    """Evaluate trained models using user-selected corruption settings."""
    X_test = trained["X_test"]
    y_test = trained["y_test"]

    corrupted = corrupt_sensor_data(
        X_test,
        noise_level=noise_level,
        missing_rate=missing_rate,
        drift_level=drift_level,
        random_state=random_state,
    )

    models = {
        "Baseline": trained["baseline"],
        "Noise-Aware": trained["noise_aware"],
    }

    records = []

    for model_name, model in models.items():
        predictions = model.predict(corrupted[FEATURES])

        records.append(
            {
                "model": model_name,
                "accuracy": accuracy_score(y_test, predictions),
                "macro_f1": f1_score(
                    y_test,
                    predictions,
                    average="macro",
                    zero_division=0,
                ),
            }
        )

    return pd.DataFrame(records)


def evaluate_models(save_outputs: bool = True) -> pd.DataFrame:
    """Evaluate both models under the five predefined scenarios."""
    trained = train_models()
    X_test = trained["X_test"]
    y_test = trained["y_test"]

    models = {
        "Baseline": trained["baseline"],
        "Noise-Aware": trained["noise_aware"],
    }

    records = []
    predictions_by_scenario = {}

    for scenario_index, (scenario, settings) in enumerate(
        SCENARIOS.items()
    ):
        corrupted = corrupt_sensor_data(
            X_test,
            random_state=100 + scenario_index,
            **settings,
        )

        predictions_by_scenario[scenario] = {}

        for model_name, model in models.items():
            predictions = model.predict(corrupted[FEATURES])

            records.append(
                {
                    "scenario": scenario,
                    "model": model_name,
                    "accuracy": accuracy_score(y_test, predictions),
                    "macro_f1": f1_score(
                        y_test,
                        predictions,
                        average="macro",
                        zero_division=0,
                    ),
                }
            )

            predictions_by_scenario[scenario][model_name] = predictions

    results = pd.DataFrame(records)

    # Compare each model's accuracy with its own clean-data accuracy.
    clean_accuracy = results.loc[
        results["scenario"] == "Clean"
    ].set_index("model")["accuracy"]

    results["accuracy_drop_percentage_points"] = results.apply(
        lambda row: (
            clean_accuracy[row["model"]] - row["accuracy"]
        ) * 100,
        axis=1,
    )

    if save_outputs:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)

        results.to_csv(RESULTS_DIR / "metrics.csv", index=False)

        # Save the fixed-scenario accuracy comparison.
        accuracy_pivot = results.pivot(
            index="scenario",
            columns="model",
            values="accuracy",
        ).reindex(SCENARIOS.keys())

        ax = accuracy_pivot.plot(
            kind="bar",
            figsize=(10, 5),
            ylim=(0, 1),
            rot=20,
        )
        ax.set_title("Model Accuracy Under Sensor Corruption")
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Evaluation scenario")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            FIGURES_DIR / "accuracy_comparison.png",
            dpi=160,
        )
        plt.close()

        # Save confusion matrices for combined corruption.
        for model_name, predictions in predictions_by_scenario[
            "Combined"
        ].items():
            display = ConfusionMatrixDisplay.from_predictions(
                y_test,
                predictions,
                labels=sorted(y_test.unique()),
                xticks_rotation=30,
                colorbar=False,
            )
            display.ax_.set_title(
                f"{model_name}: Combined Corruption"
            )
            display.figure_.tight_layout()

            safe_name = model_name.lower().replace("-", "_")

            display.figure_.savefig(
                FIGURES_DIR / f"confusion_matrix_{safe_name}.png",
                dpi=160,
            )
            plt.close(display.figure_)

    return results


if __name__ == "__main__":
    evaluation = evaluate_models()

    print("\nEvaluation metrics:")
    print(
        evaluation[
            [
                "scenario",
                "model",
                "accuracy",
                "macro_f1",
                "accuracy_drop_percentage_points",
            ]
        ].round(4).to_string(index=False)
    )

    print("\nSaved metrics to results/metrics.csv")
    print("Saved plots to results/figures/")
