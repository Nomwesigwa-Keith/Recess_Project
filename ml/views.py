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

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'soil_moisture_model.pkl')

@csrf_exempt
def predict_soil_moisture(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST requests are allowed.'}, status=405)
    try:
        data = json.loads(request.body.decode('utf-8'))
        location = data.get('location')
        temperature = data.get('temperature_celsius')
        humidity = data.get('humidity_percent')
        # Remove crop_type logic, only use soil moisture threshold
        SOIL_MOISTURE_THRESHOLD = 30.0  # Example threshold, adjust as needed
        if location is None or temperature is None or humidity is None:
            return JsonResponse({'error': 'Missing required fields: location, temperature_celsius, humidity_percent.'}, status=400)
        # Load model and preprocessors
        with open(MODEL_PATH, 'rb') as f:
            bundle = pickle.load(f)
        model = bundle['model']
        scaler = bundle['scaler']
        label_encoders = bundle['label_encoders']
        feature_cols = bundle['feature_cols']
        # Encode location
        le = label_encoders['location']
        try:
            location_encoded = le.transform([location])[0]
        except ValueError:
            return JsonResponse({'error': f'Unknown location: {location}'}, status=400)
        # Prepare features
        X = np.array([[location_encoded, float(temperature), float(humidity)]])
        X_scaled = scaler.transform(X)
        # Predict
        pred = model.predict(X_scaled)[0]
        recommendation = None
        if pred < SOIL_MOISTURE_THRESHOLD:
            recommendation = f"Irrigation needed! Predicted soil moisture ({pred:.2f}%) is below the threshold ({SOIL_MOISTURE_THRESHOLD}%)."
        else:
            recommendation = f"No irrigation needed. Predicted soil moisture ({pred:.2f}%) is above the threshold ({SOIL_MOISTURE_THRESHOLD}%)."

        # Create notifications for ALL users (admin, farmer, technician)
        # Get all users with profiles
        users_with_profiles = User.objects.filter(userprofile__isnull=False)
        
        for user in users_with_profiles:
            try:
                profile = user.userprofile
                # Create role-specific notification messages
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
                # Skip users with profile issues
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
