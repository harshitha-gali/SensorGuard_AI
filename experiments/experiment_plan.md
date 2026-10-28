# SensorGuard AI — Experiment Plan

## 1. Research Question

Does training a Random Forest classifier with noise-augmented synthetic sensor data improve classification performance under simulated random noise, missing sensor readings, fixed sensor offset, and combined corruption compared with a baseline model trained on clean data?

## 2. Dataset

- **Source:** Programmatically generated synthetic sensor data.
- **Initial dataset size:** 4,000 samples.
- **Features:** Vibration, temperature, pressure, and rotational speed.
- **Target:** `fault_type`.
- **Classes:** Normal, Overheating, Excessive_Vibration, and Pressure_Anomaly.
- **Class distribution:** 1,000 samples per class.
- **Train/test split:** 3,200 training samples and 800 test samples.

The readings are simulated and do not represent measurements from real industrial equipment.

## 3. Models

### 3.1 Baseline Model

Train a Random Forest classifier using clean training data.

### 3.2 Noise-Aware Model

Train the same classifier configuration using the clean training data plus two copies augmented with simulated random noise.

Both models use the same stratified train/test split and are evaluated on the same held-out test samples.

## 4. Test Conditions

The predefined experiment evaluates five scenarios:

1. **Clean:** Original held-out test features.
2. **Random noise:** Random fluctuations with a standard deviation proportional to each feature's standard deviation.
3. **Missing values:** A selected fraction of individual feature readings is replaced with missing values.
4. **Fixed sensor offset:** A constant positive shift proportional to each feature's standard deviation.
5. **Combined corruption:** Random noise, missing readings, and fixed sensor offset applied together.

The fixed-offset implementation does not simulate progressively increasing drift over time.

## 5. Predefined Corruption Settings

| Scenario | Noise level | Missing rate | Offset level |
|---|---:|---:|---:|
| Clean | 0.00 | 0% | 0.00 |
| Random noise | 0.30 | 0% | 0.00 |
| Missing values | 0.00 | 15% | 0.00 |
| Fixed sensor offset | 0.00 | 0% | 0.75 |
| Combined corruption | 0.30 | 15% | 0.75 |

These values are simulation parameters, not measured industrial failure rates.

## 6. Evaluation Metrics

- **Accuracy:** Fraction of test samples classified correctly.
- **Macro F1-score:** Unweighted average of the per-class F1-scores.
- **Confusion matrix:** Comparison of true and predicted classes.
- **Accuracy degradation:** Difference between a model's clean-condition accuracy and its corrupted-condition accuracy, expressed in percentage points.
- **Accuracy difference:** Noise-Aware accuracy minus Baseline accuracy for the same scenario.

## 7. Experimental Controls

- Use fixed random seeds in the predefined experiment.
- Split the data before training-data augmentation.
- Keep the held-out test set separate from model training.
- Evaluate both models on the same test samples and labels.
- Fit preprocessing steps on training data only.
- Report observed results without assuming that the Noise-Aware model will always perform better.
- Clearly distinguish synthetic results from real-world validation.

## 8. Current Results

The predefined experiment results are stored in `results/metrics.csv`.

| Scenario | Baseline Accuracy | Noise-Aware Accuracy |
|---|---:|---:|
| Clean | 98.50% | 98.50% |
| Random noise | 94.50% | 96.13% |
| Missing values | 88.00% | 87.88% |
| Fixed sensor offset | 63.75% | 75.00% |
| Combined corruption | 53.38% | 61.25% |

The Noise-Aware model performed better under random noise, fixed sensor offset, and combined corruption in this run. It performed slightly worse under missing values alone.

These results come from one predefined run with a fixed random seed. They do not establish statistical significance or generalization to real industrial equipment.

## 9. Implementation and Testing Status

The following project components have been implemented:

- Synthetic sensor-data generation.
- Sensor corruption functions.
- Baseline and Noise-Aware Random Forest training.
- Predefined scenario evaluation and metric export.
- Accuracy comparison and confusion-matrix figures.
- Interactive Streamlit dashboard with user-selected corruption settings.
- Automated tests for data generation, corruption, model training, and evaluation.

The latest confirmed automated test run reported 18 passing tests.

Interactive evaluations train a fresh pair of models and do not overwrite the original predefined results.

## 10. Limitations

- The dataset and corruption mechanisms are synthetic.
- The simulated fault classes use deliberately designed feature patterns.
- The fixed-offset simulation does not represent progressive sensor drift.
- Only selected corruption types and severities have been evaluated.
- The reported comparison uses a fixed random seed.
- Results have not been validated using real industrial sensor data.

The findings demonstrate behaviour under controlled simulations, not proven industrial reliability.

## 11. Future Experiments

1. Repeat the predefined experiments across multiple random seeds.
2. Measure variability in accuracy and macro F1-score across runs.
3. Investigate augmentation methods specifically for missing sensor readings.
4. Implement and evaluate time-dependent drift as a separate corruption scenario.
5. Explore additional corruption types and severity levels.
6. Evaluate on suitable real-world sensor datasets if available.
7. Document each additional experiment and compare it with the current baseline.
