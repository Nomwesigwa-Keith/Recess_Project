from django.db import models

class Location(models.Model):
    name = models.CharField(max_length=100)
    latitude = models.FloatField(blank=True, null=True)
    longitude = models.FloatField(blank=True, null=True)

    def __str__(self):
        return self.name

class CropType(models.Model):
    name = models.CharField(max_length=100)
    description = models.TextField(blank=True)

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

    def __str__(self):
        return f"{self.location} - {self.moisture}% at {self.recorded_at.strftime('%Y-%m-%d %H:%M')}"
