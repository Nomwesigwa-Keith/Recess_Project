from django import forms
from .models import SoilMoistureRecord, Location, CropType

class SoilMoistureRecordForm(forms.ModelForm):
    class Meta:
        model = SoilMoistureRecord
        fields = ['location', 'crop_type', 'moisture', 'temperature', 'humidity', 'notes']

class SoilMoistureCSVUploadForm(forms.Form):
    csv_file = forms.FileField(
        label='CSV File',
        help_text='Upload a CSV file with soil moisture data',
        widget=forms.FileInput(attrs={'class': 'form-control', 'accept': '.csv'})
    )

class MLModelTrainingForm(forms.Form):
    dataset_file = forms.FileField(
        label='Training Dataset (CSV)',
        help_text='Upload a CSV file with soil moisture training data',
        widget=forms.FileInput(attrs={'class': 'form-control', 'accept': '.csv'})
    )
    
    model_name = forms.CharField(
        label='Model Name',
        max_length=100,
        initial='soil_moisture_model',
        help_text='Name for the trained model',
        widget=forms.TextInput(attrs={'class': 'form-control'})
    )
    
    test_size = forms.FloatField(
        label='Test Size',
        min_value=0.1,
        max_value=0.5,
        initial=0.2,
        help_text='Percentage of data to use for testing (0.1-0.5)',
        widget=forms.NumberInput(attrs={'class': 'form-control', 'step': '0.1'})
    )
    
    random_state = forms.IntegerField(
        label='Random State',
        min_value=0,
        max_value=1000,
        initial=42,
        help_text='Random seed for reproducible results',
        widget=forms.NumberInput(attrs={'class': 'form-control'})
    )
    
    retrain_existing = forms.BooleanField(
        label='Retrain Existing Model',
        required=False,
        initial=True,
        help_text='Replace the existing model with the new one',
        widget=forms.CheckboxInput(attrs={'class': 'form-check-input'})
    )

class SoilMoistureFilterForm(forms.Form):
    location = forms.ModelChoiceField(queryset=Location.objects.all(), required=False, label='Location')
    crop_type = forms.ModelChoiceField(queryset=CropType.objects.all(), required=False, label='Crop Type')
    start_date = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}), label='Start Date')
    end_date = forms.DateField(required=False, widget=forms.DateInput(attrs={'type': 'date'}), label='End Date')
