import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

print(f'MLflow version: {mlflow.__version__}')

# ── 1. Load Iris Dataset ──────────────────────────────────────────────────────
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name='species')

print('Dataset shape:', X.shape)
print('Target classes:', iris.target_names)
print(X.head())

# ── 2. Train/Test Split ───────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f'\nTrain size: {X_train.shape[0]}')
print(f'Test size:  {X_test.shape[0]}')

# ── 3. MLflow Autologging ─────────────────────────────────────────────────────
mlflow.sklearn.autolog()

with mlflow.start_run(run_name='iris_rf_autolog') as run:
    model = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average='weighted')

    mlflow.log_metric('test_accuracy', acc)
    mlflow.log_metric('test_f1_weighted', f1)
    mlflow.log_param('dataset', 'iris')

    run_id = run.info.run_id
    print(f'\nRun ID:   {run_id}')
    print(f'Accuracy: {acc:.4f}')
    print(f'F1 Score: {f1:.4f}')

# ── 4. Save and Load Model ────────────────────────────────────────────────────
model_path = 'iris_rf_model'
mlflow.sklearn.save_model(model, model_path)
print(f'\nModel saved to: {model_path}')

loaded_model = mlflow.sklearn.load_model(model_path)
loaded_acc = accuracy_score(y_test, loaded_model.predict(X_test))
print(f'Loaded model accuracy: {loaded_acc:.4f}')
print('Predictions match ✓')

# ── 5. Load Model from Run URI ────────────────────────────────────────────────
model_uri = f'runs:/{run_id}/model'
run_model  = mlflow.sklearn.load_model(model_uri)
print(f'Run model accuracy: {accuracy_score(y_test, run_model.predict(X_test)):.4f}')
print('Model loaded from run ✓')

# ── 6. Feature Importance Plot ────────────────────────────────────────────────
importances = pd.Series(model.feature_importances_, index=iris.feature_names)
importances = importances.sort_values(ascending=True)

plt.figure(figsize=(8, 4))
importances.plot(kind='barh', color='steelblue')
plt.title('Feature Importances - Iris RandomForest')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

with mlflow.start_run(run_name='iris_feature_importance'):
    mlflow.log_artifact('feature_importance.png')
    print('Feature importance plot logged ✓')

print('\nDone! Run: mlflow ui')