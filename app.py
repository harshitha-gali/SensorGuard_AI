
"""SensorGuard AI — Interactive robustness evaluation dashboard."""

from pathlib import Path

import pandas as pd
import streamlit as st

from src.evaluate import evaluate_custom_corruption
from src.train import train_models


# --------------------------------------------------
# Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="SensorGuard AI",
    page_icon="🛡️",
    layout="wide",
)

PROJECT_DIR = Path(__file__).resolve().parent
METRICS_FILE = PROJECT_DIR / "results" / "metrics.csv"
FIGURES_DIR = PROJECT_DIR / "results" / "figures"


# --------------------------------------------------
# Dashboard heading
# --------------------------------------------------

st.title("🛡️ SensorGuard AI")
st.subheader("Robustness Evaluation of Machine-Learning Models")

st.write(
    "Explore how machine-learning models respond to simulated "
    "noise, missing sensor readings, and systematic sensor drift."
)

st.info(
    "This project uses synthetic sensor data. Its results do not "
    "establish reliability on real industrial equipment."
)


# --------------------------------------------------
# Existing experiment results
# --------------------------------------------------

st.header("Original Experiment Results")

if not METRICS_FILE.exists():
    st.error(
        "The saved metrics file was not found. "
        "Run the evaluation script before opening the dashboard."
    )
    st.stop()

results = pd.read_csv(METRICS_FILE)

required_columns = {
    "scenario",
    "model",
    "accuracy",
    "macro_f1",
    "accuracy_drop_percentage_points",
}

if not required_columns.issubset(results.columns):
    st.error("The saved metrics file is missing required columns.")
    st.stop()

clean_baseline = results.loc[
    (results["scenario"] == "Clean")
    & (results["model"] == "Baseline"),
    "accuracy",
].iloc[0]

clean_noise_aware = results.loc[
    (results["scenario"] == "Clean")
    & (results["model"] == "Noise-Aware"),
    "accuracy",
].iloc[0]

combined_noise_aware = results.loc[
    (results["scenario"] == "Combined")
    & (results["model"] == "Noise-Aware"),
    "accuracy",
].iloc[0]

col1, col2, col3 = st.columns(3)

col1.metric("Baseline — Clean Accuracy", f"{clean_baseline:.2%}")
col2.metric("Noise-Aware — Clean Accuracy", f"{clean_noise_aware:.2%}")
col3.metric(
    "Noise-Aware — Combined Accuracy",
    f"{combined_noise_aware:.2%}",
)

st.dataframe(
    results,
    use_container_width=True,
    hide_index=True,
)

st.header("Original Model Accuracy Comparison")

accuracy_chart = results.pivot(
    index="scenario",
    columns="model",
    values="accuracy",
).reindex(
    ["Clean", "Noise", "Missing Values", "Sensor Drift", "Combined"]
)

st.bar_chart(accuracy_chart, y_label="Accuracy")

st.caption(
    "These are the results from the original, predefined evaluation scenarios."
)


# --------------------------------------------------
# Original saved figures
# --------------------------------------------------

st.header("Original Confusion Matrices")

col1, col2 = st.columns(2)

with col1:
    baseline_image = FIGURES_DIR / "confusion_matrix_baseline.png"

    if baseline_image.exists():
        st.image(
            str(baseline_image),
            caption="Baseline — Combined Corruption",
            use_container_width=True,
        )
    else:
        st.warning("Baseline confusion matrix was not found.")

with col2:
    noise_aware_image = (
        FIGURES_DIR / "confusion_matrix_noise_aware.png"
    )

    if noise_aware_image.exists():
        st.image(
            str(noise_aware_image),
            caption="Noise-Aware — Combined Corruption",
            use_container_width=True,
        )
    else:
        st.warning("Noise-aware confusion matrix was not found.")

accuracy_figure = FIGURES_DIR / "accuracy_comparison.png"

if accuracy_figure.exists():
    st.image(
        str(accuracy_figure),
        caption="Original fixed-scenario experiment",
        use_container_width=True,
    )


# --------------------------------------------------
# Interactive experiment
# --------------------------------------------------

st.divider()
st.header("Interactive Sensor Corruption Experiment")

st.write(
    "Adjust the corruption settings below, then click the button "
    "to evaluate both models on the same held-out test samples."
)

noise_level = st.slider(
    "Random noise level",
    min_value=0.0,
    max_value=1.0,
    value=0.30,
    step=0.05,
    help="Controls random noise relative to each feature's standard deviation.",
)

missing_rate = st.slider(
    "Missing sensor readings",
    min_value=0.0,
    max_value=0.50,
    value=0.15,
    step=0.05,
    format="%.2f",
    help="Fraction of individual sensor readings replaced with missing values.",
)

drift_level = st.slider(
    "Sensor drift level",
    min_value=0.0,
    max_value=1.50,
    value=0.75,
    step=0.05,
    help="Controls the systematic shift relative to each feature's standard deviation.",
)

st.caption(
    "A zero setting disables that corruption type. "
    "A missing rate of 0.15 means approximately 15% of feature readings "
    "are masked, not 15% of complete rows."
)

if st.button("Run Interactive Evaluation", type="primary"):
    with st.spinner("Training models and evaluating sensor corruption..."):
        trained = train_models(save_models=False)

        interactive_results = evaluate_custom_corruption(
            trained=trained,
            noise_level=noise_level,
            missing_rate=missing_rate,
            drift_level=drift_level,
            random_state=42,
        )

    st.session_state["interactive_results"] = interactive_results
    st.session_state["interactive_settings"] = {
        "noise_level": noise_level,
        "missing_rate": missing_rate,
        "drift_level": drift_level,
    }

if "interactive_results" in st.session_state:
    interactive_results = st.session_state["interactive_results"]
    settings = st.session_state["interactive_settings"]

    st.subheader("Interactive Evaluation Results")

    st.write(
        f"**Settings:** noise = {settings['noise_level']:.2f}, "
        f"missing readings = {settings['missing_rate']:.0%}, "
        f"drift = {settings['drift_level']:.2f}"
    )

    display_results = interactive_results.copy()
    display_results["accuracy"] = display_results["accuracy"].map(
        lambda value: f"{value:.2%}"
    )
    display_results["macro_f1"] = display_results["macro_f1"].map(
        lambda value: f"{value:.2%}"
    )

    st.dataframe(
        display_results,
        use_container_width=True,
        hide_index=True,
    )

    comparison = interactive_results.set_index("model")[
        ["accuracy", "macro_f1"]
    ]

    st.bar_chart(comparison, y_label="Score")

    baseline_accuracy = interactive_results.loc[
        interactive_results["model"] == "Baseline",
        "accuracy",
    ].iloc[0]

    noise_aware_accuracy = interactive_results.loc[
        interactive_results["model"] == "Noise-Aware",
        "accuracy",
    ].iloc[0]

    difference = (noise_aware_accuracy - baseline_accuracy) * 100

    st.metric(
        "Noise-Aware accuracy difference vs Baseline",
        f"{difference:+.2f} percentage points",
    )

    st.caption(
        "This comparison uses the selected corruption settings and a newly "
        "trained pair of models. It does not overwrite the original saved "
        "metrics or figures."
    )


# --------------------------------------------------
# Interpretation and limitations
# --------------------------------------------------

st.divider()
st.header("Interpretation and Limitations")

st.markdown(
    """
    - Noise-aware training may improve performance under some corruption types.
    - Missing values, noise, and drift can affect the models differently.
    - No model is guaranteed to perform better in every condition.
    - All experiments use synthetic sensor readings and simulated corruption.
    - Real-world industrial performance has not been validated.
    """
)
