# ml/preprocessing.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib


class DataPreprocessor:
    def __init__(self):
        # Define numerical and categorical features
        self.numerical_features = [
            "soil_moisture_percent",
            "temperature_celsius",
            "humidity_percent",
            "battery_voltage",
        ]
        self.categorical_features = ["location"]
        self.target = "status"

        # Create preprocessing pipelines
        self.numerical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        self.categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )

        self.preprocessor = ColumnTransformer(
            [
                ("num", self.numerical_pipeline, self.numerical_features),
                ("cat", self.categorical_pipeline, self.categorical_features),
            ]
        )

    def fit(self, df):
        """Fit preprocessing on training data"""
        # Additional feature engineering
        df = self._feature_engineering(df)

        # Fit preprocessor
        self.preprocessor.fit(df)

        # Fit label encoder for target
        self.label_encoder = LabelEncoder()
        if self.target in df.columns:
            self.label_encoder.fit(df[self.target])

        return self

    def transform(self, df):
        """Transform data using fitted preprocessor"""
        # Feature engineering
        df = self._feature_engineering(df)

        # Transform features
        features = self.preprocessor.transform(df)

        # Transform target if present
        target = None
        if self.target in df.columns:
            target = self.label_encoder.transform(df[self.target])

        return features, target

    def _feature_engineering(self, df):
        """Create additional features"""
        df = df.copy()

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Time-based features
        df["hour"] = df["timestamp"].dt.hour
        df["day_of_week"] = df["timestamp"].dt.dayofweek
        df["is_daytime"] = df["hour"].between(6, 18).astype(int)

        # Environmental interactions
        df["temp_humidity_ratio"] = df["temperature_celsius"] / (
            df["humidity_percent"] + 0.01
        )
        df["moisture_temp_interaction"] = (
            df["soil_moisture_percent"] * df["temperature_celsius"]
        )

        # Battery status categories
        df["battery_status"] = pd.cut(
            df["battery_voltage"],
            bins=[0, 3.3, 3.7, 4.2],
            labels=["low", "medium", "high"],
        )

        return df

    def save(self, filepath):
        """Save preprocessor to file"""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath):
        """Load preprocessor from file"""
        return joblib.load(filepath)
