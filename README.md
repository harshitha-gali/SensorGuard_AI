# SensorGuard AI

### Robustness Evaluation of Machine-Learning Models Under Simulated Sensor Noise

## 1. Overview

SensorGuard AI is a machine-learning project that investigates how sensor corruption affects the performance of sensor-based fault classification models.

The project generates synthetic industrial-style sensor readings, simulates different types of sensor corruption, and compares a baseline model with a noise-aware model trained using noise-augmented data.

An interactive Streamlit dashboard allows users to adjust corruption settings and evaluate both models under selected conditions.

**Important:** The dataset is synthetic. The results demonstrate model behaviour in a controlled simulation and do not establish reliability on real industrial equipment.

## 2. Problem Statement

Sensor readings can be affected by random noise, missing measurements, and systematic sensor drift. These disturbances may reduce the accuracy of machine-learning systems that classify equipment conditions.

This project investigates whether noise-augmented training can improve the robustness of a machine-learning classifier under simulated sensor corruption.

## 3. Objectives

- Generate reproducible synthetic sensor data.
- Simulate random noise, missing sensor readings, and sensor drift.
- Train and compare baseline and noise-aware classifiers.
- Evaluate performance using accuracy and macro F1-score.
- Visualize model performance across predefined and user-selected corruption scenarios.
- Provide automated tests for the core project functions.

## 4. Sensor Features and Classes

The synthetic dataset contains 4,000 samples, with four sensor features and four balanced target classes.

**Sensor features**
- Vibration
- Temperature
- Pressure
- Rotational speed

**Classification labels**
- Normal
- Overheating
- Excessive Vibration
- Pressure Anomaly

The dataset is generated programmatically with a fixed random seed for reproducibility.

## 5. Methodology

1. Generate the synthetic sensor dataset.
2. Split the dataset into stratified training and test sets.
3. Train a baseline Random Forest classifier on clean training data.
4. Train a noise-aware Random Forest classifier using clean data and noise-augmented copies.
5. Apply simulated corruption to the held-out test data.
6. Evaluate both models using accuracy and macro F1-score.
7. Compare performance using tables, charts, and confusion matrices.
8. Explore custom corruption settings using the interactive dashboard.

Both models use a preprocessing pipeline with median imputation, feature scaling, and a Random Forest classifier.

## 6. Simulated Corruption Types

- **Random noise:** Adds random variation relative to each feature's standard deviation.
- **Missing sensor readings:** Replaces a proportion of individual sensor readings with missing values.
- **Sensor drift:** Applies a systematic shift relative to each feature's standard deviation.
- **Combined corruption:** Applies multiple corruption types together.

These are controlled simulations, not measurements of actual sensor failure behaviour.

## 7. Experimental Results

Results from the predefined evaluation scenarios:

| Scenario | Baseline Accuracy | Noise-Aware Accuracy |
|---|---:|---:|
| Clean | 98.50% | 98.50% |
| Random noise | 94.50% | 96.13% |
| Missing values | 88.00% | 87.88% |
| Sensor drift | 63.75% | 75.00% |
| Combined corruption | 53.38% | 61.25% |

### Key observations

- Both models achieved 98.50% accuracy on the clean test data.
- Noise-aware training improved accuracy under simulated random noise and sensor drift.
- Under combined corruption, the noise-aware model achieved 61.25% accuracy compared with 53.38% for the baseline.
- The noise-aware model performed slightly worse under the missing-values-only scenario.

The results indicate that noise augmentation can help under some simulated corruption conditions, but it does not guarantee better performance for every type of corruption.

## 8. Interactive Dashboard

The Streamlit dashboard provides:

- Summary metrics from the predefined experiments.
- A comparison of baseline and noise-aware model performance.
- Confusion matrices for the combined-corruption scenario.
- Adjustable controls for random noise, missing readings, and sensor drift.
- Interactive evaluation results with accuracy and macro F1-score comparisons.

Interactive evaluations do not overwrite the original saved experiment results.

## 9. Project Structure

```text
SensorGuard_AI/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data_generation.py
│   ├── corruption.py
│   ├── train.py
│   └── evaluate.py
├── experiments/
│   └── experiment_plan.md
├── results/
│   ├── metrics.csv
│   └── figures/
├── docs/
│   ├── problem_statement.md
│   ├── methodology.md
│   ├── results_analysis.md
│   └── screenshots/
├── tests/
│   ├── test_data_generation.py
│   ├── test_corruption.py
│   ├── test_train.py
│   └── test_evaluate.py
└── models/
    ├── baseline_model.joblib
    └── noise_aware_model.joblib
```

## 10. Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- pip

### Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### Run the automated tests

From the project root:

```powershell
python -m pytest -v
```

### Generate the predefined evaluation results

```powershell
python -m src.evaluate
```

This runs the fixed evaluation scenarios and saves the metrics and figures.

### Launch the dashboard

```powershell
python -m streamlit run app.py
```

Open the local URL displayed in the terminal.

## 11. Testing

The project currently includes 18 automated tests covering data generation, sensor corruption, model training, and evaluation.

**Latest recorded test result:** 18 passed.

Run the test suite again after code changes to verify the current state.

## 12. Limitations and Future Work

- The dataset is synthetic and does not capture the full complexity of real industrial sensor data.
- The corruption mechanisms are simulated and may not represent actual sensor failures.
- The evaluation covers a limited set of corruption types and intensity levels.
- Results have not been validated on real machinery or an external industrial dataset.
- Further work could investigate real sensor datasets, unseen corruption patterns, severity-based evaluation, and additional robustness metrics.

## 13. Technologies Used

- Python
- NumPy
- pandas
- scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- pytest
- joblib

## 14. Reproducibility and Responsible Reporting

The project uses fixed random seeds in its data-generation and experiment code to support reproducibility. Exact results may depend on the software environment and the selected experiment settings.

All reported performance values should be interpreted as results from synthetic data and simulated corruption, not as evidence of validated industrial reliability.
