from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework import status
from .models import SoilMoistureRecord, Location, CropType
from .serializers import SoilMoistureRecordSerializer
import json
import logging

logger = logging.getLogger(__name__)

def latest_soil_moisture(request):
    # Get the latest soil moisture record
    latest = SoilMoistureRecord.objects.order_by('-recorded_at').first()
    if latest:
        data = {
            'moisture': latest.moisture,
            'location': str(latest.location),
            'crop_type': str(latest.crop_type) if latest.crop_type else None,
            'recorded_at': latest.recorded_at.strftime('%Y-%m-%d %H:%M:%S'),
            'temperature': latest.temperature,
            'humidity': latest.humidity,
            'notes': latest.notes,
        }
    else:
        data = {'moisture': None, 'location': None, 'crop_type': None, 'recorded_at': None, 'temperature': None, 'humidity': None, 'notes': None}
    return JsonResponse(data)

@csrf_exempt
@require_http_methods(["POST"])
def iot_soil_moisture_reading(request):
    """
    IoT Device API endpoint for submitting soil moisture readings
    Accepts POST requests with JSON data from IoT devices
    
    Expected JSON format:
    {
        "device_id": "sensor_001",
        "location_id": 1,
        "moisture": 45.5,
        "temperature": 25.3,
        "humidity": 60.2,
        "crop_type_id": 1,  // optional
        "notes": "Automated reading from IoT sensor"  // optional
    }
    """
    try:
        # Parse JSON data
        data = json.loads(request.body.decode('utf-8'))
        
        # Validate required fields
        required_fields = ['device_id', 'location_id', 'moisture']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return JsonResponse({
                'error': f'Missing required fields: {", ".join(missing_fields)}',
                'required_fields': required_fields
            }, status=400)
        
        # Validate data types and ranges
        try:
            moisture = float(data['moisture'])
            if not (0 <= moisture <= 100):
                return JsonResponse({
                    'error': 'Moisture value must be between 0 and 100'
                }, status=400)
        except (ValueError, TypeError):
            return JsonResponse({
                'error': 'Invalid moisture value. Must be a number.'
            }, status=400)
        
        # Validate location exists
        try:
            location = Location.objects.get(id=data['location_id'])
        except Location.DoesNotExist:
            return JsonResponse({
                'error': f'Location with ID {data["location_id"]} does not exist'
            }, status=400)
        
        # Validate crop_type if provided
        crop_type = None
        if 'crop_type_id' in data and data['crop_type_id']:
            try:
                crop_type = CropType.objects.get(id=data['crop_type_id'])
            except CropType.DoesNotExist:
                return JsonResponse({
                    'error': f'Crop type with ID {data["crop_type_id"]} does not exist'
                }, status=400)
        
        # Validate temperature and humidity if provided
        temperature = None
        humidity = None
        
        if 'temperature' in data and data['temperature'] is not None:
            try:
                temperature = float(data['temperature'])
                if not (-50 <= temperature <= 100):
                    return JsonResponse({
                        'error': 'Temperature value must be between -50 and 100'
                    }, status=400)
            except (ValueError, TypeError):
                return JsonResponse({
                    'error': 'Invalid temperature value. Must be a number.'
                }, status=400)
        
        if 'humidity' in data and data['humidity'] is not None:
            try:
                humidity = float(data['humidity'])
                if not (0 <= humidity <= 100):
                    return JsonResponse({
                        'error': 'Humidity value must be between 0 and 100'
                    }, status=400)
            except (ValueError, TypeError):
                return JsonResponse({
                    'error': 'Invalid humidity value. Must be a number.'
                }, status=400)
        
        # Create the soil moisture record
        record = SoilMoistureRecord.objects.create(
            location=location,
            crop_type=crop_type,
            moisture=moisture,
            temperature=temperature,
            humidity=humidity,
            sensor_id=data.get('device_id'),
            sensor_type=data.get('sensor_type', 'soil_moisture'),  # Default to soil_moisture
            notes=data.get('notes', f'Automated reading from IoT device: {data["device_id"]}')
        )
        
        # Log the successful reading
        logger.info(f'IoT device {data["device_id"]} submitted reading: moisture={moisture}%, location={location.name}')
        
        # Return success response
        return JsonResponse({
            'success': True,
            'message': 'Soil moisture reading recorded successfully',
            'record_id': record.id,
            'recorded_at': record.recorded_at.strftime('%Y-%m-%d %H:%M:%S'),
            'data': {
                'moisture': record.moisture,
                'temperature': record.temperature,
                'humidity': record.humidity,
                'location': str(record.location),
                'crop_type': str(record.crop_type) if record.crop_type else None,
                'sensor_id': record.sensor_id,
                'sensor_type': record.sensor_type
            }
        }, status=201)
        
    except json.JSONDecodeError:
        return JsonResponse({
            'error': 'Invalid JSON format in request body'
        }, status=400)
    except Exception as e:
        logger.error(f'Error processing IoT reading: {str(e)}')
        return JsonResponse({
            'error': 'Internal server error while processing reading'
        }, status=500)

@api_view(['GET'])
def iot_api_status(request):
    """
    API status endpoint for IoT devices to check connectivity
    """
    return JsonResponse({
        'status': 'online',
        'message': 'IoT API is operational',
        'endpoints': {
            'submit_reading': '/soil/api/iot-reading/',
            'latest_reading': '/soil/api/latest-soil-moisture/',
            'api_status': '/soil/api/iot-status/'
        }
    })
