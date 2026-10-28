# SensorGuard AI — Problem Statement

## 1. Project Title

**SensorGuard AI: Robustness Evaluation of Machine-Learning Models Under Simulated Sensor Noise**

## 2. Introduction

Machine-learning models can classify operating conditions using sensor measurements. In industrial environments, sensor readings may be affected by random noise, missing values, calibration offsets, or gradual sensor drift.

These disturbances can change the input data distribution and reduce a model's ability to classify operating conditions correctly.

SensorGuard AI investigates how simulated sensor corruption affects fault classification and whether training with noise-augmented data can improve performance under selected corruption conditions.

## 3. Problem Statement

A machine-learning classifier trained on clean sensor readings may experience reduced classification performance when its input readings become noisy, incomplete, or systematically shifted.

The problem addressed in this project is to measure the effect of simulated sensor corruption and compare a baseline classifier with a classifier trained using noise-augmented data.

The project uses synthetic sensor readings and controlled experiments. It does not claim to diagnose actual industrial equipment.

## 4. Aim

To evaluate the robustness of machine-learning-based sensor fault classification under simulated sensor corruption and investigate the effect of noise-augmented training.

## 5. Objectives

1. Generate a reproducible synthetic dataset containing four simulated operating conditions.
2. Use vibration, temperature, pressure, and rotational speed as input features.
3. Simulate random noise, missing sensor readings, and a fixed sensor offset.
4. Train a baseline Random Forest classifier using clean training data.
5. Train a noise-aware Random Forest classifier using clean and noise-augmented training data.
6. Compare both models using accuracy, macro F1-score, and confusion matrices.
7. Develop an interactive Streamlit dashboard to visualize evaluation results under predefined and user-selected corruption settings.
8. Document the methodology, results, limitations, and reproducibility steps.

## 6. Research Question

Does training a Random Forest classifier with noise-augmented synthetic sensor data improve its classification performance under simulated random noise, missing values, fixed sensor offset, and combined corruption compared with a baseline model trained on clean data?

## 7. Scope

The project focuses on:

- Synthetic industrial-style sensor-data generation.
- Four simulated operating classes: Normal, Overheating, Excessive Vibration, and Pressure Anomaly.
- Controlled sensor-corruption experiments.
- Comparison of baseline and noise-aware machine-learning models.
- Interactive visualization of model performance.

The current implementation does not include physical sensor integration, live industrial monitoring, or validation using real machinery data. Its fixed-offset simulation does not model drift that progressively increases over time.

## 8. Expected Outcome

The expected outcome is a working experimental application that measures and visualizes how the two classifiers perform under different simulated sensor conditions.

Noise-aware training may improve performance under some conditions but may not improve every corruption scenario. Conclusions are therefore based on measured experimental results rather than assumed improvements.

## 9. Dataset Limitation

All sensor readings are generated synthetically for this project. The results demonstrate model behaviour under the selected simulations and do not establish real-world industrial reliability.
