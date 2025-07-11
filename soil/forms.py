from django import forms
from .models import SoilMoistureRecord

class SoilMoistureRecordForm(forms.ModelForm):
    class Meta:
        model = SoilMoistureRecord
        fields = ['location', 'crop_type', 'moisture', 'temperature', 'humidity', 'notes']
