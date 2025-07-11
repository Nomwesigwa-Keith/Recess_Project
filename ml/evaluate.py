# ml/evaluate.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from train import load_data
import joblib


def evaluate_model():
    """Evaluate model performance and generate visualizations"""
    # Load data and artifacts
    X, y, preprocessor = load_data("ml/data/soil_moisture_dataset.csv")
    model = joblib.load("ml/models/moisture_model.pkl")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Make predictions
    y_pred = model.predict(X_test)

    # Generate classification report
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv("ml/evaluation/classification_report.csv")

    # Generate confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        xticklabels=preprocessor.label_encoder.classes_,
        yticklabels=preprocessor.label_encoder.classes_,
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig("ml/evaluation/confusion_matrix.png")
    plt.close()

    # Generate class distribution plot
    plt.figure(figsize=(10, 6))
    pd.Series(y).value_counts().plot(kind="bar")
    plt.title("Class Distribution")
    plt.xlabel("Status")
    plt.ylabel("Count")
    plt.savefig("ml/evaluation/class_distribution.png")
    plt.close()

    print("Evaluation complete. Reports and visualizations saved.")


if __name__ == "__main__":
    evaluate_model()
