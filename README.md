# MLflow Experiment Tracking Lab - Iris Dataset
## IE7374 MLOps | Lab Assignment 5

This lab demonstrates MLflow experiment tracking and model management using the **Iris classification dataset** as a modification from the original Wine Quality dataset.

---

## 📋 Table of Contents
- [Overview](#overview)
- [Modifications from Original Lab](#modifications-from-original-lab)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Output Structure](#output-structure)
- [Key Enhancements](#key-enhancements)
- [Performance Metrics](#performance-metrics)
- [Visualizations](#visualizations)
- [Results Export](#results-export)

---

## 🎯 Overview
This lab covers the full MLflow experiment tracking lifecycle:
- **Autologging** with RandomForestClassifier
- **Manual logging** of parameters, metrics, and artifacts
- **Hyperparameter sweep** across 5 values of regularization strength C
- **Model registration**, saving, and loading
- **Model serving** with sample inference

The lab uses the **Iris dataset** (3-class flower classification) to demonstrate these concepts on a clean, dependency-free dataset built into sklearn.

---

## 🔄 Modifications from Original Lab

| | Original Lab | This Submission |
|---|---|---|
| Dataset | Wine Quality (CSV files) | Iris (sklearn built-in) |
| Task | Binary Classification | Multi-class Classification (3 classes) |
| Models | RandomForestClassifier only | RandomForest + LogisticRegression |
| Metrics | AUC score only | Accuracy, F1, Precision, Recall |
| Runs | Single baseline run | Hyperparameter sweep (5 runs) |
| Artifacts | Model only | Confusion matrix, feature importance plot, hyperparameter sweep plot, classification report, feature importance JSON, class names JSON |

---

## ✨ Features

### Core MLflow Concepts
- `mlflow.sklearn.autolog()` — automatic parameter/metric/model logging
- `mlflow.log_param()` — manual hyperparameter tracking
- `mlflow.log_metric()` — performance metric logging
- `mlflow.log_artifact()` — plots and files as artifacts
- `mlflow.sklearn.log_model()` — model persistence
- `mlflow.sklearn.load_model()` — model reloading from run URI
- `mlflow.search_runs()` — programmatic run querying

### Enhanced Features Added
**1. Multiple Models**
- RandomForestClassifier (starter.py)
- LogisticRegression with StandardScaler (linear_regression.py)

**2. Comprehensive Metrics**
- Accuracy: Overall classification correctness
- F1 Score (weighted): Balanced precision-recall metric
- Precision (weighted): Positive predictive value
- Recall (weighted): Sensitivity across all classes

**3. Hyperparameter Sweep**
- 5 runs with C values: [0.01, 0.1, 1.0, 10.0, 100.0]
- All runs tracked and queryable in MLflow UI
- Best C identified programmatically

**4. Artifacts Logged**
- `confusion_matrix.png` — visual classification performance
- `hyperparameter_sweep.png` — Accuracy & F1 vs C plot
- `feature_importance.png` — feature importance bar chart
- `feature_importance.json` — structured feature importance data
- `classification_report.txt` — full sklearn classification report
- `class_names.json` — target class metadata

**5. Model Serving**
- Model logged with input example
- Sample inference demonstrated
- Serve command printed for real-time inference

---

## 📦 Requirements
```
mlflow>=2.9.0
scikit-learn>=1.2.0
pandas>=1.5.0
numpy>=1.23.0
matplotlib>=3.6.0
seaborn>=0.12.0
ipykernel>=6.0.0
jupyterlab>=3.0.0
cloudpickle>=2.0.0
requests>=2.28.0
```

---

## 🚀 Installation
```bash
# 1. Clone the repo
git clone https://github.com/sushmaramesh16/MLOPS-LAB-5-MLFLOW.git
cd MLOPS-LAB-5-MLFLOW

# 2. Create virtual environment
python3 -m venv mlops_lab5_env
source mlops_lab5_env/bin/activate  # Mac/Linux
# mlops_lab5_env\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt
```

---

## 💻 Usage

Run each script in order:
```bash
# Step 1: Autologging + save/load model
python3 starter.py

# Step 2: Manual logging + hyperparameter sweep
python3 linear_regression.py

# Step 3: Custom artifacts + model serving
python3 serving.py

# Step 4: View all experiments in MLflow UI
mlflow ui
# Open http://127.0.0.1:5000
```

---

## 📁 Output Structure
```
MLOPS_LAB5/
├── starter.py                    # Autolog + save/load model
├── linear_regression.py          # Manual logging + hyperparam sweep
├── serving.py                    # Custom artifacts + serving
├── requirements.txt
├── README.md
├── confusion_matrix.png          # Generated artifact
├── hyperparameter_sweep.png      # Generated artifact
├── feature_importance.png        # Generated artifact
├── feature_importance.json       # Generated artifact
├── classification_report.txt     # Generated artifact
├── class_names.json              # Generated artifact
└── mlruns/                       # MLflow tracking directory
```

---

## 📊 Performance Metrics

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| RandomForest (autolog) | 0.9667 | 0.9666 |
| LogisticRegression (C=1.0) | 0.9333 | 0.9333 |
| LogisticRegression (C=10.0) | 1.0000 | 1.0000 |
| RandomForest (serving) | 0.9000 | — |

---

## 📈 Visualizations

**1. Feature Importance** — petal length and width are the strongest predictors for Iris classification

**2. Confusion Matrix** — shows near-perfect classification with minor versicolor/virginica overlap (expected behavior)

**3. Hyperparameter Sweep** — accuracy and F1 both improve as C increases, plateauing at C=10.0

---

## 🔗 MLflow Experiments Tracked

| Experiment | Runs |
|---|---|
| `iris_logistic_regression` | 6 runs (1 baseline + 5 sweep) |
| `iris_serving` | 1 run |
| Default experiment | 2 runs (autolog + feature importance) |

---

## 🎓 Learning Objectives

- Understand MLflow tracking concepts: experiments, runs, params, metrics, artifacts
- Compare autologging vs manual logging approaches
- Use MLflow UI to compare multiple runs visually
- Save and reload models using MLflow model URIs
- Serve a model for real-time inference

---

## 🔗 Resources
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Original Lab Repository](https://github.com/raminmohammadi/MLOps/tree/main/Labs/Experiment_Tracking_Labs/Mlflow_Labs/Lab1)
- [Scikit-learn Iris Dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html)

---

## 👤 Author
**Sushma Ramesh** — MS Data Science, Northeastern University  
