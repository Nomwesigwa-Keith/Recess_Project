from rest_framework import serializers
from .models import SoilMoistureRecord

class SoilMoistureRecordSerializer(serializers.ModelSerializer):
    class Meta:
        model = SoilMoistureRecord
        fields = ['location', 'crop_type', 'moisture', 'temperature', 'humidity', 'notes', 'sensor_id', 'sensor_type'] 