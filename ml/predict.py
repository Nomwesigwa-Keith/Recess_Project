# ml/predict.py
import pandas as pd
import numpy as np
import joblib
from typing import Dict, Any


class SoilMoisturePredictor:
    def __init__(
        self,
        model_path="ml/models/moisture_model.pkl",
        preprocessor_path="ml/models/preprocessor.pkl",
    ):
        """Initialize predictor with trained model and preprocessor"""
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)

    def predict(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Make prediction on new data"""
        # Convert input to DataFrame
        input_df = pd.DataFrame([input_data])

        # Preprocess input
        X, _ = self.preprocessor.transform(input_df)

        # Make prediction
        encoded_pred = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        # Get confidence and predicted class
        confidence = np.max(probabilities)
        status = self.preprocessor.label_encoder.inverse_transform([encoded_pred])[0]

        # Determine irrigation action
        action = self._get_irrigation_action(status)

        return {
            "status": status,
            "confidence": float(confidence),
            "irrigation_action": action,
            "probabilities": dict(
                zip(
                    self.preprocessor.label_encoder.classes_,
                    [float(p) for p in probabilities],
                )
            ),
        }

    def _get_irrigation_action(self, status: str) -> str:
        """Determine irrigation action based on status"""
        if status in ["Dry", "Critical Low"]:
            return "Irrigate"
        elif status in ["Wet", "Critical High"]:
            return "Reduce Irrigation"
        else:
            return "Maintain Current Irrigation"
