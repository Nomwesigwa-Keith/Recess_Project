from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse, HttpResponse
from soil.models import CropType
from accounts.models import Notification, UserProfile
from django.contrib.auth.models import User
import pickle
import numpy as np
import os
import json
from ml.models import PredictionHistory

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'soil_moisture_model.pkl')

@csrf_exempt
def predict_soil_moisture(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)
    try:
        data = json.loads(request.body.decode('utf-8'))
        # Required features
        location = data.get('location')
        temperature = data.get('temperature_celsius')
        humidity = data.get('humidity_percent')
        rainfall_24h = data.get('rainfall_24h', 0.0)
        days_since_irrigation = data.get('days_since_irrigation', 0.0)
        soil_type = data.get('soil_type', 'Loamy')
        timestamp = data.get('timestamp')  # Should be ISO string if provided
        # Remove crop_type logic, only use soil moisture threshold
        SOIL_MOISTURE_THRESHOLD = 40.0  # Updated threshold as requested
        if location is None or temperature is None or humidity is None:
            return JsonResponse({'error': 'Missing required fields: location, temperature_celsius, humidity_percent.'}, status=400)
        # Load model and preprocessors
        with open(MODEL_PATH, 'rb') as f:
            bundle = pickle.load(f)
        model = bundle['model']
        scaler = bundle['scaler']
        label_encoders = bundle['label_encoders']
        feature_cols = bundle['feature_cols']
        # Encode categorical features
        try:
            location_encoded = label_encoders['location'].transform([location])[0]
        except Exception:
            return JsonResponse({'error': f'Unknown location: {location}'}, status=400)
        try:
            soil_type_encoded = label_encoders['soil_type'].transform([soil_type])[0]
        except Exception:
            soil_type_encoded = 0  # fallback if unknown
        # Handle timestamp-based features
        import datetime
        if timestamp:
            try:
                dt = datetime.datetime.fromisoformat(timestamp)
            except Exception:
                dt = datetime.datetime.now()
        else:
            dt = datetime.datetime.now()
        hour = dt.hour
        day_of_week = dt.weekday()
        month = dt.month
        # Derived features
        temp_humidity_interaction = float(temperature) * float(humidity)
        temp_squared = float(temperature) ** 2
        humidity_squared = float(humidity) ** 2
        # Assemble features in correct order
        X = np.array([[location_encoded,
                      float(temperature),
                      float(humidity),
                      float(rainfall_24h),
                      float(days_since_irrigation),
                      soil_type_encoded,
                      hour,
                      day_of_week,
                      month,
                      temp_humidity_interaction,
                      temp_squared,
                      humidity_squared]])
        X_scaled = scaler.transform(X)
        # Predict
        pred = model.predict(X_scaled)[0]
        recommendation = None
        if pred < SOIL_MOISTURE_THRESHOLD:
            recommendation = f"Irrigation needed! Predicted soil moisture ({pred:.2f}%) is below the threshold ({SOIL_MOISTURE_THRESHOLD}%)."
        else:
            recommendation = f"No irrigation needed. Predicted soil moisture ({pred:.2f}%) is above the threshold ({SOIL_MOISTURE_THRESHOLD}%)."
        # Save prediction to history
        user = None
        if request.user.is_authenticated:
            user = request.user
        PredictionHistory.objects.create(
            user=user,
            location=location,
            temperature_celsius=temperature,
            humidity_percent=humidity,
            predicted_moisture=pred
        )
        # Create notifications for ALL users (admin, farmer, technician)
        users_with_profiles = User.objects.filter(userprofile__isnull=False)
        for user in users_with_profiles:
            try:
                profile = user.userprofile
                if profile.role == 'farmer':
                    notification_message = f"🌾 {recommendation} Location: {location}"
                elif profile.role == 'technician':
                    notification_message = f"🔧 System Alert: {recommendation} Location: {location}"
                elif profile.role == 'admin':
                    notification_message = f"📊 Admin Alert: {recommendation} Location: {location}"
                else:
                    notification_message = f"System Alert: {recommendation} Location: {location}"
                Notification.objects.create(
                    user=user,
                    message=notification_message
                )
            except Exception as e:
                continue
        return JsonResponse({
            'predicted_soil_moisture_percent': float(pred),
            'recommendation': recommendation
        })
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)

def ml_predictions_page(request):
    # Load valid locations from the trained model's label encoder
    locations = []
    try:
        with open(MODEL_PATH, 'rb') as f:
            bundle = pickle.load(f)
        le = bundle['label_encoders']['location']
        locations = list(le.classes_)
    except Exception:
        pass  # If model not found, show empty dropdown
    crop_types = CropType.objects.all()
    return render(request, 'ml_predictions.html', {'locations': locations, 'crop_types': crop_types})
