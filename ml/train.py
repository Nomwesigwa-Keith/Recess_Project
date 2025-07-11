# ml/train.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from preprocessing import DataPreprocessor
import joblib
import matplotlib.pyplot as plt
import numpy as np


def load_data(filepath):
    """Load and preprocess data"""
    df = pd.read_csv(filepath)
    preprocessor = DataPreprocessor()
    preprocessor.fit(df)
    X, y = preprocessor.transform(df)
    return X, y, preprocessor


def train_model(X, y):
    """Train and tune Random Forest model"""
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Define model and hyperparameters
    model = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5, 10],
    }

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=model, param_grid=param_grid, cv=5, scoring="accuracy", n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    # Get best model
    best_model = grid_search.best_estimator_

    # Evaluate
    y_pred = best_model.predict(X_test)
    print("Best Parameters:", grid_search.best_params_)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("\nAccuracy:", accuracy_score(y_test, y_pred))

    # Plot feature importance
    plot_feature_importance(best_model, preprocessor)

    return best_model


def plot_feature_importance(model, preprocessor):
    """Plot feature importance"""
    # Get feature names from preprocessor
    num_features = preprocessor.numerical_features
    cat_features = (
        preprocessor.preprocessor.named_transformers_["cat"]
        .named_steps["onehot"]
        .get_feature_names_out(input_features=preprocessor.categorical_features)
    )
    all_features = np.concatenate([num_features, cat_features])

    # Plot importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(12, 6))
    plt.title("Feature Importance")
    plt.bar(range(len(indices)), importances[indices], align="center")
    plt.xticks(range(len(indices)), [all_features[i] for i in indices], rotation=90)
    plt.tight_layout()
    plt.savefig("ml/models/feature_importance.png")
    plt.close()


if __name__ == "__main__":
    # Load and preprocess data
    X, y, preprocessor = load_data("ml/data/soil_moisture_dataset.csv")

    # Train model
    model = train_model(X, y)

    # Save artifacts
    joblib.dump(model, "ml/models/moisture_model.pkl")
    preprocessor.save("ml/models/preprocessor.pkl")

    print("Model training complete and artifacts saved.")
