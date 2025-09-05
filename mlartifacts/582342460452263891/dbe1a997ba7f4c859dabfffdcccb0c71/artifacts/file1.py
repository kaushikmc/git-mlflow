import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set MLflow tracking server
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Create/Use experiment
mlflow.set_experiment("Sample1")

# Load dataset
wine = load_wine()
X = wine.data
y = wine.target

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42
)

# Define RF params
max_depth = 15
n_estimators = 10

with mlflow.start_run() as run:
    # Model training
    rf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators, random_state=42)
    rf.fit(X_train, y_train)

    # Predictions
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log params and metrics
    mlflow.log_param("max_depth", max_depth)
    mlflow.log_param("n_estimators", n_estimators)
    mlflow.log_metric("accuracy", accuracy)

    # Log tags (per-run)
    mlflow.set_tag("Author", "Kaushik")
    mlflow.set_tag("Project", "Wine Classification")

    # Create and save confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=wine.target_names, yticklabels=wine.target_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title("Confusion Matrix")
    plt.tight_layout()

    # Save the plot
    plot_path = "Confusion-matrix.png"
    plt.savefig(plot_path)
    plt.close()

    # Log artifacts (plot + this script)
    mlflow.log_artifact(plot_path)
    if os.path.exists(__file__):  # avoid errors in interactive sessions
        mlflow.log_artifact(__file__)

    # Log the trained model
    mlflow.sklearn.log_model(rf, artifact_path="Random-Forest-Classifier")

    print(f"Run {run.info.run_id} completed.")
    print("Accuracy:", accuracy)
