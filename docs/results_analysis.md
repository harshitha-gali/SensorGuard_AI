# SensorGuard AI — Results Analysis

## 1. Experiment Overview

Two Random Forest classifiers were evaluated on the same held-out test set
of 800 synthetic sensor samples.

- **Baseline model:** Trained on clean training data.
- **Noise-Aware model:** Trained on clean training data augmented with
  simulated random sensor noise.
- Both models used the same stratified train/test split.
- Evaluation scenarios included clean readings, random noise, missing
  values, sensor drift, and combined corruption.

All sensor readings are synthetic. They do not represent measurements
collected from real industrial equipment.

## 2. Experimental Results

The following results were recorded in `results/metrics.csv`.

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

Percentages are rounded to two decimal places. Accuracy drop is calculated
relative to each model's own clean-condition accuracy.

## 3. Comparison of Model Accuracy

Accuracy difference is calculated as:

**Accuracy difference = Noise-Aware accuracy − Baseline accuracy**

| Scenario | Accuracy Difference |
|---|---:|
| Clean | 0.00 percentage points |
| Noise | +1.63 percentage points |
| Missing Values | -0.13 percentage points |
| Sensor Drift | +11.25 percentage points |
| Combined | +7.88 percentage points |

Positive values indicate higher accuracy for the Noise-Aware model.
Negative values indicate lower accuracy.

## 4. Findings

### 4.1 Random Noise

The Baseline model achieved 94.50% accuracy, while the Noise-Aware model
achieved 96.13%.

The improvement was approximately 1.63 percentage points under the
tested random-noise condition.

### 4.2 Missing Values

The Baseline model achieved 88.00% accuracy, compared with 87.88% for
the Noise-Aware model.

The difference was small, and noise augmentation did not improve
accuracy under this condition. Random-noise augmentation alone may not
provide sufficient robustness to missing sensor readings.

### 4.3 Sensor Drift

The Baseline model achieved 63.75% accuracy, while the Noise-Aware model
achieved 75.00%.

The improvement was 11.25 percentage points, the largest observed
accuracy gain among the tested corrupted scenarios.

This suggests that random-noise augmentation may also help under the
particular drift simulation used in this experiment. Additional
experiments are needed to determine how consistently this effect occurs.

### 4.4 Combined Corruption

The Baseline model achieved 53.38% accuracy, compared with 61.25% for
the Noise-Aware model.

The improvement was approximately 7.88 percentage points. However,
both models performed worse under combined corruption than under
random noise alone.

Simultaneous sensor disturbances remained challenging for both models.

## 5. Interpretation

The results suggest that noise-aware training can improve classification
performance under some simulated corruption conditions.

The largest observed improvement occurred under sensor drift, followed
by combined corruption. Under missing values, performance was nearly
identical, with a slight accuracy disadvantage for the Noise-Aware model.

These findings are specific to the synthetic dataset, selected
corruption severities, model configuration, and random seed used in
the experiment. They do not establish that noise-aware training will
improve robustness under all conditions.

## 6. Limitations

- All sensor readings are synthetic.
- The simulated fault classes use deliberately designed feature patterns.
- Only selected corruption types and severity levels were evaluated.
- The current comparison uses a fixed random seed.
- Results have not been validated on real industrial data.
- Repeatability and statistical variation across multiple random seeds
  have not yet been assessed.

The results demonstrate model behaviour under controlled simulation,
not validated reliability on real industrial equipment.

## 7. Automated Testing

The project uses pytest to test its main data and machine-learning
components.

**Latest confirmed test result:** 18 passed, 0 failed.

| Component | Tests | Result |
|---|---:|---|
| Data generation | 6 | Passed |
| Sensor corruption | 6 | Passed |
| Model training | 3 | Passed |
| Model evaluation | 3 | Passed |
| **Total** | **18** | **All passed** |

The tests cover dataset structure, class balance, reproducibility,
invalid inputs, corruption behaviour, model training, prediction
generation, evaluation scenarios, metric ranges, and accuracy-drop
calculations.

These tests establish software behaviour for the tested cases. They do
not establish that the synthetic dataset accurately represents real
industrial sensor behaviour.

## 8. Interactive Dashboard

The Streamlit dashboard provides:

- Summary metrics from the predefined experiments.
- A comparison of Baseline and Noise-Aware model performance.
- Confusion matrices for combined corruption.
- Adjustable random-noise, missing-reading, and sensor-drift settings.
- Interactive evaluation results with accuracy and macro F1-score comparisons.

Interactive evaluations train a fresh pair of models and do not
overwrite the original saved experiment results.

## 9. Future Work

1. Repeat experiments using multiple random seeds to assess variability.
2. Investigate dedicated augmentation strategies for missing readings
   and sensor drift.
3. Evaluate additional corruption types and severity levels.
4. Investigate performance on suitable real-world sensor datasets.
5. Compare additional model configurations and robustness metrics.
6. Document further experiments and compare their results with the
   current baseline.

## 10. Conclusion

In this synthetic-data experiment, noise-aware training improved
accuracy under random noise, sensor drift, and combined corruption,
but did not improve accuracy under missing values alone.

These findings provide a starting point for investigating robustness
in sensor-based fault classification. Real-world validation and
additional experiments are required before drawing conclusions about
industrial deployment.