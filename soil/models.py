from django.db import models
from django.contrib.auth.models import User

class Location(models.Model):
    name = models.CharField(max_length=100)
    latitude = models.FloatField(blank=True, null=True)
    longitude = models.FloatField(blank=True, null=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='locations', null=True, blank=True)

    def __str__(self):
        return self.name

class CropType(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)
    water_requirement = models.FloatField(
        help_text="Crop water requirement in mm/day or other unit", blank=True, null=True)
    moisture_threshold = models.FloatField(
        help_text="Minimum soil moisture % before irrigation is recommended", blank=True, null=True)
    created_by = models.ForeignKey(User, on_delete=models.CASCADE, related_name='crop_types', null=True, blank=True)

    def __str__(self):
        return self.name

class SoilMoistureRecord(models.Model):
    location = models.ForeignKey(Location, on_delete=models.CASCADE)
    crop_type = models.ForeignKey(CropType, on_delete=models.SET_NULL, null=True, blank=True)
    moisture = models.FloatField()
    recorded_at = models.DateTimeField(auto_now_add=True)
    temperature = models.FloatField(blank=True, null=True)
    humidity = models.FloatField(blank=True, null=True)
    notes = models.TextField(blank=True)
    # New fields for sensor support
    sensor_id = models.CharField(max_length=100, blank=True, null=True, help_text="Unique identifier for the sensor")
    sensor_type = models.CharField(max_length=50, blank=True, null=True, help_text="Type of sensor (e.g., 'soil_moisture', 'temperature', 'humidity')")

    def __str__(self):
        return f"{self.location} - {self.moisture}% at {self.recorded_at.strftime('%Y-%m-%d %H:%M')}"
