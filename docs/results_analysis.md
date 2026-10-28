
# SensorGuard AI — Results Analysis

## 1. Experiment Overview

Two Random Forest classifiers were compared using a held-out test set
of 800 synthetic sensor samples.

- **Baseline model:** Trained on clean training data.
- **Noise-aware model:** Trained on clean training data augmented with
  simulated random sensor noise.
- Both models were evaluated using the same held-out test samples.
- Evaluation conditions included clean readings, random noise, missing
  values, sensor drift, and combined corruption.

All sensor readings are synthetic and do not represent measurements
collected from real industrial equipment.

## 2. Results

The following table summarizes the results currently available for
the four corrupted test conditions.

| Scenario | Model | Accuracy | Macro F1-score | Accuracy drop (percentage points) |
|---|---|---:|---:|---:|
| Clean | Baseline | 98.50% | 98.50% | 0.00 |
| Clean | Noise-Aware | 98.50% | 98.51% | 0.00 |
| Noise | Baseline | 94.50% | 94.48% | 4.00 |
| Noise | Noise-Aware | 96.13% | 96.12% | 2.38 |
| Missing Values | Baseline | 88.00% | 88.59% | 10.50 |
| Missing Values | Noise-Aware | 87.88% | 88.47% | 10.63 |
| Sensor Drift | Baseline | 63.75% | 62.52% | 34.75 |
| Sensor Drift | Noise-Aware | 75.00% | 73.07% | 23.50 |
| Combined | Baseline | 53.38% | 51.87% | 45.13 |
| Combined | Noise-Aware | 61.25% | 59.99% | 37.25 |

Clean-condition metrics will be added after checking the saved evaluation
results in `results/metrics.csv`.

## 3. Comparison of Model Accuracy

Accuracy differences are calculated as:

**Accuracy difference = Noise-aware accuracy − Baseline accuracy**

| Scenario | Accuracy Difference |
|---|---:|
| Noise | +1.62 percentage points |
| Missing Values | -0.12 percentage points |
| Sensor Drift | +11.25 percentage points |
| Combined | +7.88 percentage points |

Positive values indicate higher accuracy for the noise-aware model.
Negative values indicate lower accuracy.

## 4. Findings

### 4.1 Random Noise

The baseline model achieved 94.50% accuracy, while the noise-aware model
achieved 96.12%.

This represents an improvement of 1.62 percentage points under the
tested random-noise condition.

### 4.2 Missing Values

The baseline model achieved 88.00% accuracy, compared with 87.88% for
the noise-aware model.

The difference is small, and the noise-aware model did not improve
accuracy under this condition. Random-noise augmentation alone may not
provide sufficient robustness to missing sensor readings.

### 4.3 Sensor Drift

The baseline model achieved 63.75% accuracy, while the noise-aware model
achieved 75.00%.

The improvement of 11.25 percentage points is the largest observed
accuracy gain among the corrupted test conditions.

This result suggests that training with random-noise augmentation may
also help under the particular drift simulation used in this experiment.
Further experiments are needed to determine how consistent this effect is.

### 4.4 Combined Corruption

The baseline model achieved 53.37% accuracy, compared with 61.25% for
the noise-aware model.

The noise-aware model improved accuracy by 7.88 percentage points.
However, both models performed worse under combined corruption than
under random noise alone.

This indicates that simultaneous sensor problems remain challenging
for both models.

## 5. Interpretation

The current results suggest that noise-aware training can improve
classification performance under some simulated corruption conditions.

The largest observed improvement occurred under sensor drift, followed
by combined corruption. Under missing values, performance was nearly
identical for both models.

These findings are specific to the synthetic dataset, selected
corruption severities, model configuration, and random seed used in
the experiment. They do not establish that noise-aware training will
improve robustness under all conditions.

## 6. Limitations

- All sensor readings are synthetic.
- The simulated operating classes use deliberately designed feature
  patterns.
- Only selected corruption types and severity levels were evaluated.
- The current comparison uses a fixed random seed.
- The results have not yet been validated using real industrial data.
- Further experiments are needed to assess repeatability and statistical
  variation.

Consequently, the findings demonstrate behavior under controlled
simulations rather than validated industrial reliability.

## 7. Automated Testing

The synthetic sensor data generation module was tested using `pytest`.

**Test results:** 6 passed, 0 failed.

The tests verified:
- Expected dataset dimensions and column names.
- Presence of all four fault classes.
- Balanced class distribution.
- Reproducibility with a fixed random seed.
- Correct handling of an invalid sample count.

These tests validate the behavior of the data generation module. They do not establish that the synthetic data accurately represents real industrial sensor behavior.

## 7. Automated Testing

The project uses pytest to verify the behavior of its main data and machine-learning components.

**Latest test run:** 18 passed, 0 failed, in 10.87 seconds.

| Component | Tests | Result |
|---|---:|---|
| Data generation | 6 | Passed |
| Sensor corruption | 6 | Passed |
| Model training | 3 | Passed |
| Model evaluation | 3 | Passed |
| **Total** | **18** | **All passed** |

The tests cover dataset structure, class balance, reproducibility, invalid inputs, corruption behavior, model training, prediction generation, evaluation scenarios, metric ranges, and clean-data accuracy-drop calculations.

These tests establish basic software correctness for the tested cases. They do not establish real-world industrial reliability or validate the synthetic data against physical sensor measurements.

## 8. Next Steps

1. Retrieve and include the clean-condition metrics.
2. Complete the Streamlit dashboard and visualize the results.
3. Add automated tests for data generation and corruption.
4. Repeat the experiments with multiple random seeds.
5. Investigate dedicated augmentation strategies for missing values
   and sensor drift.
6. Document any additional experiments and compare their results with
   the baseline experiment.
7. Capture screenshots and complete the project README.
