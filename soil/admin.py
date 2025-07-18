from django.contrib import admin
from .models import Location, CropType, SoilMoistureRecord

@admin.register(Location)
class LocationAdmin(admin.ModelAdmin):
    list_display = ("name", "latitude", "longitude")

@admin.register(CropType)
class CropTypeAdmin(admin.ModelAdmin):
    list_display = ("name",)

@admin.register(SoilMoistureRecord)
class SoilMoistureRecordAdmin(admin.ModelAdmin):
    list_display = ("location", "moisture", "recorded_at", "temperature", "humidity", "crop_type", "sensor_id", "sensor_type")
    list_filter = ("location", "crop_type", "recorded_at", "sensor_type")
    search_fields = ("location__name", "notes", "sensor_id")
