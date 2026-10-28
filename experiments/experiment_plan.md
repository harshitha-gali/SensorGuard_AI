
# SensorGuard AI — Experiment Plan

## 1. Research Question

Can training a machine-learning model with corrupted sensor data improve its
performance when evaluated on noisy, incomplete, or drifted sensor readings?

## 2. Dataset

- Source: Programmatically generated synthetic sensor data.
- Initial dataset size: 4,000 samples.
- Features: vibration, temperature, pressure, rotational speed.
- Target: fault_type.
- Classes: Normal, Overheating, Excessive_Vibration, Pressure_Anomaly.
- Class distribution: 1,000 samples per class.

These readings are simulated and do not represent measurements from real
industrial equipment.

## 3. Models

### Baseline model
Train a classifier using clean training data.

### Noise-aware model
Train the same type of classifier using clean training data augmented with
simulated sensor noise.

Both models will be evaluated on the same held-out test samples.

## 4. Test Conditions

1. Clean sensor readings.
2. Random sensor noise.
3. Missing sensor readings.
4. Systematic sensor drift.
5. Combined corruption.

## 5. Evaluation Metrics

- Accuracy
- Macro F1-score
- Confusion matrix
- Performance degradation relative to clean evaluation

Macro F1-score gives each class equal weight when calculating the average.

## 6. Experimental Controls

- Use fixed random seeds for reproducibility.
- Split the data before training.
- Keep the test set separate from model training.
- Apply the same test samples to both models.
- Fit preprocessing steps using training data only.
- Report actual results without assuming the noise-aware model will always win.

## 7. Limitations

The dataset and corruptions are synthetic. Findings demonstrate behavior under
the chosen simulations and should not be treated as proof of reliability on
real industrial equipment.
