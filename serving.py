import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
import json
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# ── 1. Load Data ──────────────────────────────────────────────────────────────
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target, name="species")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 2. Train and Log ──────────────────────────────────────────────────────────
mlflow.set_experiment("iris_serving")

with mlflow.start_run(run_name="iris_rf_serving") as run:
    n_estimators, max_depth = 50, 5

    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("dataset", "iris")
    mlflow.log_param("n_features", X.shape[1])
    mlflow.log_param("n_classes", len(iris.target_names))

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)

    report = classification_report(y_test, y_pred, target_names=iris.target_names)
    with open("classification_report.txt", "w") as f:
        f.write("Iris RandomForest Classification Report\n")
        f.write("=" * 45 + "\n")
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    feature_importance = dict(zip(iris.feature_names, model.feature_importances_.tolist()))
    with open("feature_importance.json", "w") as f:
        json.dump(feature_importance, f, indent=2)
    mlflow.log_artifact("feature_importance.json")

    with open("class_names.json", "w") as f:
        json.dump({"classes": iris.target_names.tolist()}, f, indent=2)
    mlflow.log_artifact("class_names.json")

    mlflow.sklearn.log_model(model, "iris_rf_model", input_example=X_test.iloc[:3])

    run_id = run.info.run_id
    print(f"Run ID:   {run_id}")
    print(f"Accuracy: {acc:.4f}")
    print(f"\nTo serve:\n  mlflow models serve --env-manager=local -m runs:/{run_id}/iris_rf_model -h 127.0.0.1 -p 5001")

# ── 3. Sample Inference ───────────────────────────────────────────────────────
loaded = mlflow.sklearn.load_model(f"runs:/{run_id}/iris_rf_model")

sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]], columns=iris.feature_names)
pred = loaded.predict(sample)
pred_proba = loaded.predict_proba(sample)

print(f"\nSample input: {sample.values.tolist()[0]}")
print(f"Predicted class: {iris.target_names[pred[0]]}")
print(f"Probabilities: {dict(zip(iris.target_names, pred_proba[0].round(3)))}")

print("\nDone! Run: mlflow ui")