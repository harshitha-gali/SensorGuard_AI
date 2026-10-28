# SensorGuard AI — Methodology

## 1. Overview

SensorGuard AI investigates how simulated sensor corruption affects machine-learning-based fault classification and whether training with noise-augmented data improves robustness.

The project uses a synthetic dataset and compares two Random Forest classifiers under predefined and user-selected corruption conditions.

All data is synthetic. The experiment does not establish reliability on real industrial equipment.

## 2. Dataset Generation

The dataset is generated programmatically using NumPy and pandas.

It contains 4,000 samples and five columns:

| Column | Description |
|---|---|
| `vibration` | Simulated vibration measurement |
| `temperature` | Simulated machine temperature |
| `pressure` | Simulated pressure measurement |
| `rotational_speed` | Simulated rotational speed |
| `fault_type` | Target class representing the simulated operating condition |

The four target classes are:

- `Normal`
- `Overheating`
- `Excessive_Vibration`
- `Pressure_Anomaly`

Each class contains 1,000 samples in the generated dataset.

These values are synthetic and are not measurements collected from real industrial machinery.

## 3. Data Splitting

The dataset is divided into:

- Training set: 3,200 samples (80%)
- Test set: 800 samples (20%)

A stratified train/test split preserves the class proportions in both sets. A fixed random seed supports reproducibility.

The test set is separated before training-data augmentation. Augmented copies of training samples are not added to the test set.

## 4. Model Design

### 4.1 Baseline Model

The baseline model is a Random Forest classifier trained on clean training data.

It provides a reference for measuring the effects of simulated sensor corruption.

### 4.2 Noise-Aware Model

The noise-aware model uses the same classifier configuration but is trained on the clean training data plus two noisy copies of the training features.

Random noise is added relative to each feature's standard deviation. The original labels are retained because the simulated corruption represents measurement disturbance rather than a change in the assigned class.

The augmented training set contains three versions of the training data: the original clean set and two noisy copies.

Both models are evaluated using the same held-out test samples and test labels.

## 5. Preprocessing and Classification

Each model uses a scikit-learn pipeline containing:

1. Median imputation to handle missing sensor readings.
2. Standard scaling of sensor features.
3. A Random Forest classifier with 150 trees.

Preprocessing parameters are fitted using the training data for each model, not the held-out test data. This helps prevent test-data leakage.

Standard scaling is included as part of the common pipeline, although Random Forest models generally do not require feature standardization.

## 6. Sensor Corruption Scenarios

The evaluation applies corruption to copies of the held-out test features. The class labels are retained.

### 6.1 Clean Data

The original held-out test features are used without intentional corruption.

### 6.2 Random Noise

Random fluctuations are added to sensor readings. The noise standard deviation is set as a multiple of each feature's standard deviation.

For example, a noise level of `0.30` uses a noise standard deviation equal to 30% of the corresponding feature's standard deviation.

### 6.3 Missing Values

A selected fraction of individual sensor readings is replaced with missing values (`NaN`).

For example, a missing rate of `0.15` masks approximately 15% of individual feature readings, not 15% of complete rows.

The pipeline's median imputer handles missing values during prediction.

### 6.4 Sensor Drift

A constant positive offset is added to every reading of each feature. The offset is proportional to the feature's standard deviation.

For example, a drift level of `0.75` adds an offset equal to 75% of the corresponding feature's standard deviation.

**Important:** This implementation simulates a fixed sensor offset. It does not model drift that progressively increases over time.

### 6.5 Combined Corruption

Random noise, missing values, and the fixed sensor offset are applied together.

These settings are controlled simulations. They should not be interpreted as measured industrial failure rates or realistic estimates of field conditions.

## 7. Evaluation Metrics

### 7.1 Accuracy

Accuracy is the fraction of test samples classified correctly.

### 7.2 Macro F1-score

Macro F1-score calculates the F1-score for each class and averages the scores equally. It helps assess performance across the four fault classes without weighting each class by its sample count.

### 7.3 Accuracy Degradation

Accuracy degradation is measured relative to the same model's accuracy on clean test data.

Accuracy degradation (percentage points) =

(clean accuracy - corrupted accuracy) x 100

A larger positive value indicates a larger reduction in accuracy relative to the clean condition.

### 7.4 Accuracy Difference Between Models

Accuracy difference is calculated as:

Noise-Aware accuracy - Baseline accuracy

A positive value indicates that the Noise-Aware model achieved higher accuracy for the tested condition. A negative value indicates that the Baseline model achieved higher accuracy.

## 8. Experimental Procedure

1. Generate the synthetic sensor dataset.
2. Split the data into stratified training and test sets.
3. Train the Baseline model on clean training data.
4. Generate two noisy copies of the training features.
5. Train the Noise-Aware model on the augmented training set.
6. Apply each predefined corruption scenario to the same held-out test features.
7. Evaluate both models against the same test labels.
8. Record accuracy, macro F1-score, and accuracy degradation.
9. Save the predefined evaluation metrics and figures.
10. Interpret the results while documenting the limitations.

The interactive dashboard separately allows users to choose corruption levels and evaluate a freshly trained pair of models. Interactive evaluations do not overwrite the original saved experiment results.

## 9. Reproducibility and Fairness

Fixed random seeds are used in the data generation, splitting, augmentation, and predefined evaluation code to support repeatability.

The models use the same classifier type and configuration, and both are evaluated on the same held-out test samples under each scenario.

The current reported results use a fixed random seed. Experiments across multiple independent seeds would be needed to assess variability and determine how consistently the observed differences occur.

## 10. Limitations

- The dataset is synthetic and does not capture the full complexity of real industrial sensor measurements.
- The simulated fault classes use deliberately designed feature patterns.
- The corruption mechanisms represent selected mathematical disturbances rather than validated physical sensor-failure processes.
- The drift implementation applies a fixed offset rather than progressive time-dependent drift.
- Only selected corruption types and severity levels have been evaluated.
- The initial comparison uses a fixed random seed.
- Results have not been validated against real industrial equipment or an independent real-world dataset.

SensorGuard AI should therefore be interpreted as a controlled robustness experiment, not a validated industrial fault-detection system.

## 11. Future Work

1. Repeat experiments across multiple random seeds.
2. Investigate dedicated augmentation strategies for missing sensor readings.
3. Evaluate progressive and time-dependent drift simulations.
4. Explore additional corruption types and severity levels.
5. Evaluate suitable real-world sensor datasets where available.
6. Document further experiments and compare their results with the current baseline.
