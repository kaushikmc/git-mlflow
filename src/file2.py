# file: src/file1.py
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
# ---------------- Configuration ----------------
MLFLOW_TRACKING_URI = "https://dagshub.com/kaushikmc/git-mlflow.mlflow"


dagshub.init(repo_owner='kaushikmc', repo_name='git-mlflow', mlflow=True)


EXPERIMENT_NAME = "Sample1"
REGISTERED_MODEL_NAME = "WineRFModel"
# RF hyperparams
MAX_DEPTH = 15
N_ESTIMATORS = 10
TEST_SIZE = 0.10
RANDOM_STATE = 42
# ------------------------------------------------

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

# Enable autologging for sklearn (optional but helpful)
# Load data
wine = load_wine()
X = wine.data
y = wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

# Ensure working directory for artifacts (optional)
os.makedirs("artifacts", exist_ok=True)

with mlflow.start_run() as run:
    rf = RandomForestClassifier(max_depth=MAX_DEPTH, n_estimators=N_ESTIMATORS, random_state=RANDOM_STATE)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.set_tag("Author", "Kaushik")
    mlflow.set_tag("Project", "Wine Classification")

    # Log confusion matrix plot
    cm_path = "Confusion-matrix.png"
    plt.figure(figsize=(6, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues",
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # 🔑 Explicitly log the model
# NEW CODE (will work with DagsHub)
   
    run_id = run.info.run_id
    print(f"Run {run_id} completed. Accuracy: {accuracy:.6f}")
    print(f"Artifacts at: {mlflow.get_artifact_uri()}")

# ---------------- Register the model in Model Registry ----------------
# If you don't want to register, you can skip this block.
# client = MlflowClient(MLFLOW_TRACKING_URI)
# model_uri = f"runs:/{run_id}/model"  # points to the artifact we logged above

# Create registered model if not exists
# try:
#     client.create_registered_model(REGISTERED_MODEL_NAME)
# except Exception:
#     # already exists or other benign error; ignore
#     pass

# Register a new version pointing to this run's model artifact
# mv = client.create_model_version(name=REGISTERED_MODEL_NAME, source=model_uri, run_id=run_id)
# print(f"Created model version: name={REGISTERED_MODEL_NAME}, version_id={mv.version}, run_id={run_id}")
# print("You can view it in the MLflow UI under Models ->", REGISTERED_MODEL_NAME)
