
# SensorGuard AI — Methodology

## 1. Overview

SensorGuard AI investigates how simulated sensor corruption affects
machine-learning-based fault classification and whether training with
noise-augmented data improves robustness.

The project uses a synthetic dataset and a controlled comparison of two
Random Forest classifiers.

## 2. Dataset Generation

The dataset is generated programmatically using NumPy and pandas.

It contains 4,000 samples and five columns:

| Column | Description |
|---|---|
| vibration | Simulated vibration measurement |
| temperature | Simulated machine temperature |
| pressure | Simulated pressure measurement |
| rotational_speed | Simulated rotational speed in RPM |
| fault_type | Target class representing the operating condition |

The four target classes are:

- Normal
- Overheating
- Excessive_Vibration
- Pressure_Anomaly

Each class contains 1,000 samples in the generated dataset.

These values are synthetic and are not measurements from real industrial
machinery.

## 3. Data Splitting

The dataset is divided into:

- Training set: 3,200 samples (80%)
- Test set: 800 samples (20%)

A stratified split preserves the class distribution across the two sets.
The random seed is fixed to support reproducibility.

The test set is kept separate from training and noise augmentation.

## 4. Model Design

### 4.1 Baseline Model

The baseline model is a Random Forest classifier trained on the clean
training data.

It provides a reference for measuring the effect of simulated sensor
corruption.

### 4.2 Noise-Aware Model

The noise-aware model uses the same classifier configuration but is trained
on the clean training data plus two noisy copies of the training features.

Random noise is added relative to each feature's standard deviation.
The labels remain unchanged because the corruption simulates measurement
errors rather than a change in the underlying operating condition.

Both models use the same held-out test set.

## 5. Preprocessing

Each model uses a scikit-learn pipeline containing:

1. Median imputation to handle missing sensor readings.
2. Standard scaling of sensor features.
3. A Random Forest classifier with 150 trees.

The preprocessing steps are fitted during model training, not on the test
set. This helps prevent test-data leakage.

Although scaling is included for consistency, Random Forest models generally
do not require standard scaling to operate.

## 6. Sensor Corruption Scenarios

The test data is evaluated under the following conditions:

### Clean Data

The original held-out test features are used without intentional corruption.

### Random Noise

Random fluctuations are added to sensor readings. The noise standard
deviation is controlled relative to each feature's own standard deviation.

### Missing Values

A selected fraction of individual sensor readings is replaced with missing
values. The model's median imputer handles these values during prediction.

### Sensor Drift

A systematic shift is added to sensor readings relative to each feature's
standard deviation.

### Combined Corruption

Random noise, missing values, and systematic drift are applied together.

These corruption settings are controlled simulations and should not be
interpreted as measured industrial failure rates.

## 7. Evaluation Metrics

### Accuracy

Accuracy is the fraction of test samples classified correctly.

### Macro F1-score

Macro F1 calculates the F1-score for each class and averages the scores
equally. It helps assess performance across all four fault classes.

### Accuracy Degradation

Accuracy degradation is calculated relative to the same model's accuracy
on clean test data:

Accuracy degradation (percentage points) =
(clean accuracy - corrupted accuracy) × 100

A larger positive value indicates a larger reduction in accuracy.

## 8. Experimental Procedure

1. Generate the synthetic dataset.
2. Split the data into training and test sets.
3. Train the baseline model on clean training data.
4. Create noisy copies of the training features.
5. Train the noise-aware model on the augmented training data.
6. Apply each corruption scenario to the same held-out test features.
7. Evaluate both models using the same test labels.
8. Save the metrics as a CSV file.
9. Generate accuracy comparison and confusion-matrix figures.
10. Interpret the results and record the limitations.

## 9. Reproducibility and Fairness

Fixed random seeds are used for data generation, splitting, augmentation,
and the initial evaluation.

The two models use the same classifier type and configuration, and both
are evaluated on the same held-out samples.

Further experiments with multiple random seeds are needed to determine
whether the observed differences remain consistent across runs.

## 10. Limitations

- The dataset is synthetic.
- The simulated fault classes use deliberately designed feature patterns.
- Only selected corruption types and severity levels are evaluated.
- The initial results use a fixed random seed.
- Results do not establish performance on real industrial equipment.

The project should therefore be interpreted as a controlled robustness
experiment rather than a validated industrial fault-detection system.
