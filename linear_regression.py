import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix, ConfusionMatrixDisplay
)

print(f'MLflow version: {mlflow.__version__}')

# ── 1. Load Iris Dataset ──────────────────────────────────────────────────────
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print(f'Train: {X_train.shape}, Test: {X_test.shape}')

# ── 2. Baseline Run ───────────────────────────────────────────────────────────
mlflow.set_experiment('iris_logistic_regression')

with mlflow.start_run(run_name='iris_logreg_baseline') as run:
    C, max_iter, solver = 1.0, 200, 'lbfgs'

    mlflow.log_param('C', C)
    mlflow.log_param('max_iter', max_iter)
    mlflow.log_param('solver', solver)
    mlflow.log_param('dataset', 'iris')
    mlflow.log_param('scaler', 'StandardScaler')

    model = LogisticRegression(C=C, max_iter=max_iter, solver=solver, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    recall    = recall_score(y_test, y_pred, average='weighted')

    mlflow.log_metric('accuracy', acc)
    mlflow.log_metric('f1_weighted', f1)
    mlflow.log_metric('precision_weighted', precision)
    mlflow.log_metric('recall_weighted', recall)
    mlflow.sklearn.log_model(model, 'logistic_regression_model')

    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=iris.target_names)
    disp.plot(ax=ax, colorbar=False)
    ax.set_title('Confusion Matrix - Iris LogReg')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    plt.show()

    print(f'Accuracy: {acc:.4f} | F1: {f1:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}')

# ── 3. Hyperparameter Sweep ───────────────────────────────────────────────────
C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
results  = []

for C_val in C_values:
    with mlflow.start_run(run_name=f'iris_logreg_C={C_val}'):
        mlflow.log_param('C', C_val)
        mlflow.log_param('max_iter', 300)
        mlflow.log_param('solver', 'lbfgs')
        mlflow.log_param('dataset', 'iris')

        m = LogisticRegression(C=C_val, max_iter=300, solver='lbfgs', random_state=42)
        m.fit(X_train_scaled, y_train)
        preds = m.predict(X_test_scaled)

        acc_val = accuracy_score(y_test, preds)
        f1_val  = f1_score(y_test, preds, average='weighted')

        mlflow.log_metric('accuracy', acc_val)
        mlflow.log_metric('f1_weighted', f1_val)
        mlflow.sklearn.log_model(m, 'model')

        results.append({'C': C_val, 'accuracy': acc_val, 'f1': f1_val})
        print(f'C={C_val:<6} | Accuracy: {acc_val:.4f} | F1: {f1_val:.4f}')

results_df = pd.DataFrame(results)
print('\nBest C:', results_df.loc[results_df.accuracy.idxmax(), 'C'])

# ── 4. Plot Sweep Results ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
ax.semilogx(results_df['C'], results_df['accuracy'], marker='o', label='Accuracy', color='steelblue')
ax.semilogx(results_df['C'], results_df['f1'], marker='s', label='F1 (weighted)', color='coral')
ax.set_xlabel('C (Regularization)')
ax.set_ylabel('Score')
ax.set_title('Logistic Regression: Accuracy & F1 vs C on Iris')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hyperparameter_sweep.png')
plt.show()

# ── 5. Query Runs Programmatically ────────────────────────────────────────────
runs_df = mlflow.search_runs(experiment_names=['iris_logistic_regression'])
cols = ['run_id', 'params.C', 'metrics.accuracy', 'metrics.f1_weighted']
available = [c for c in cols if c in runs_df.columns]
print(runs_df[available].sort_values('metrics.accuracy', ascending=False).head(10))

print('\nDone! Run: mlflow ui')