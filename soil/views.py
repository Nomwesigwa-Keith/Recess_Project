from django.shortcuts import render, redirect
from .models import SoilMoistureRecord
from .forms import SoilMoistureRecordForm, SoilMoistureCSVUploadForm, SoilMoistureFilterForm
from .serializers import SoilMoistureRecordSerializer
import pandas as pd
from django.contrib import messages
from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.core.paginator import Paginator
from rest_framework import generics, filters
from django_filters.rest_framework import DjangoFilterBackend
from django.contrib.auth.decorators import user_passes_test
from django.shortcuts import render, redirect, get_object_or_404
from .models import Location, CropType
from django import forms
from django.db.models import Avg, Min, Max, Count
from django.utils import timezone
from django.db.models.functions import TruncDate
import csv
from django.http import HttpResponse
from django.template.loader import render_to_string
import logging
from django.contrib.auth.decorators import login_required

# Set up logging
logger = logging.getLogger(__name__)

# Decorator for staff admin (not superuser)
def staff_admin_required(view_func):
    return user_passes_test(lambda u: u.is_staff and not u.is_superuser)(view_func)

@login_required
def location_list(request):
    # Show all locations to everyone, but farmers can only edit their own
    locations = Location.objects.all()
    is_farmer_readonly = hasattr(request.user, 'userprofile') and request.user.userprofile.role == 'farmer'
    return render(request, 'admin_location_list.html', {
        'locations': locations,
        'is_farmer_readonly': is_farmer_readonly
    })

class LocationForm(forms.ModelForm):
    class Meta:
        model = Location
        fields = ['name', 'latitude', 'longitude']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'latitude': forms.NumberInput(attrs={'class': 'form-control'}),
            'longitude': forms.NumberInput(attrs={'class': 'form-control'}),
        }

@login_required
def admin_add_location(request):
    if request.method == 'POST':
        form = LocationForm(request.POST)
        if form.is_valid():
            location = form.save(commit=False)
            if hasattr(request.user, 'userprofile') and request.user.userprofile.role == 'farmer':
                location.created_by = request.user
            location.save()
            return redirect('admin_location_list')
    else:
        form = LocationForm()
    return render(request, 'admin_add_location.html', {'form': form})

@staff_admin_required
def admin_edit_location(request, location_id):
    location = get_object_or_404(Location, pk=location_id)
    if request.method == 'POST':
        form = LocationForm(request.POST, instance=location)
        if form.is_valid():
            form.save()
            return redirect('admin_location_list')
    else:
        form = LocationForm(instance=location)
    return render(request, 'admin_edit_location.html', {'form': form, 'location': location})

@staff_admin_required
def admin_delete_location(request, location_id):
    location = get_object_or_404(Location, pk=location_id)
    if request.method == 'POST':
        location.delete()
        return redirect('admin_location_list')
    return render(request, 'admin_confirm_delete_location.html', {'location': location})

@login_required
def crop_type_list(request):
    # Show all crop types to everyone, but farmers can only edit their own
    crop_types = CropType.objects.all()
    is_farmer_readonly = hasattr(request.user, 'userprofile') and request.user.userprofile.role == 'farmer'
    return render(request, 'admin_crop_type_list.html', {
        'crop_types': crop_types,
        'is_farmer_readonly': is_farmer_readonly
    })

class CropTypeForm(forms.ModelForm):
    class Meta:
        model = CropType
        fields = ['name', 'description']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control'}),
            'description': forms.Textarea(attrs={'class': 'form-control', 'rows': 3}),
        }

@login_required
def admin_add_crop_type(request):
    if request.method == 'POST':
        form = CropTypeForm(request.POST)
        if form.is_valid():
            crop_type = form.save(commit=False)
            if hasattr(request.user, 'userprofile') and request.user.userprofile.role == 'farmer':
                crop_type.created_by = request.user
            crop_type.save()
            return redirect('admin_crop_type_list')
    else:
        form = CropTypeForm()
    return render(request, 'admin_add_crop_type.html', {'form': form})

@staff_admin_required
def admin_edit_crop_type(request, crop_type_id):
    crop_type = get_object_or_404(CropType, pk=crop_type_id)
    if request.method == 'POST':
        form = CropTypeForm(request.POST, instance=crop_type)
        if form.is_valid():
            form.save()
            return redirect('admin_crop_type_list')
    else:
        form = CropTypeForm(instance=crop_type)
    return render(request, 'admin_edit_crop_type.html', {'form': form, 'crop_type': crop_type})

@staff_admin_required
def admin_delete_crop_type(request, crop_type_id):
    crop_type = get_object_or_404(CropType, pk=crop_type_id)
    if request.method == 'POST':
        crop_type.delete()
        return redirect('admin_crop_type_list')
    return render(request, 'admin_confirm_delete_crop_type.html', {'crop_type': crop_type})

def soil_list(request):
    form = SoilMoistureFilterForm(request.GET or None)
    records = SoilMoistureRecord.objects.select_related('location', 'crop_type').order_by('-recorded_at')

    # For farmers: show ALL records but in read-only mode
    # For admins/technicians: show all records with full access
    # No filtering needed - farmers can see everything but can't edit

    if form.is_valid():
        if form.cleaned_data.get('location'):
            records = records.filter(location=form.cleaned_data['location'])
        if form.cleaned_data.get('crop_type'):
            records = records.filter(crop_type=form.cleaned_data['crop_type'])
        if form.cleaned_data.get('start_date'):
            records = records.filter(recorded_at__date__gte=form.cleaned_data['start_date'])
        if form.cleaned_data.get('end_date'):
            records = records.filter(recorded_at__date__lte=form.cleaned_data['end_date'])

    paginator = Paginator(records, 10)  # Show 10 records per page
    page_number = request.GET.get('page')
    page_obj = paginator.get_page(page_number)

    # Pass a flag to the template to indicate read-only mode for farmers
    is_farmer_readonly = hasattr(request.user, 'userprofile') and request.user.userprofile.role == 'farmer'

    return render(request, 'soil/soil_list.html', {
        'form': form,
        'page_obj': page_obj,
        'records': page_obj.object_list,
        'is_farmer_readonly': is_farmer_readonly,
    })

def add_soil_record(request):
    if request.method == 'POST':
        form = SoilMoistureRecordForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('soil_list')
    else:
        form = SoilMoistureRecordForm()
    return render(request, 'soil/add_soil_record.html', {'form': form})

def upload_soil_csv(request):
    """
    Enhanced CSV upload view that handles both regular CSV upload and ML model training
    """
    from .forms import MLModelTrainingForm
    
    # Initialize forms
    csv_form = SoilMoistureCSVUploadForm()
    ml_form = MLModelTrainingForm()
    training_metrics = None
    
    if request.method == 'POST':
        # Check if this is an ML training request
        if 'dataset_file' in request.FILES:
            # Handle ML model training
            ml_form = MLModelTrainingForm(request.POST, request.FILES)
            if ml_form.is_valid():
                try:
                    # Get uploaded file
                    dataset_file = request.FILES['dataset_file']
                    model_name = ml_form.cleaned_data['model_name']
                    test_size = ml_form.cleaned_data['test_size']
                    random_state = ml_form.cleaned_data['random_state']
                    retrain_existing = ml_form.cleaned_data['retrain_existing']
                    
                    # Import ML libraries
                    import os
                    import pickle
                    import numpy as np
                    from sklearn.model_selection import train_test_split
                    from sklearn.ensemble import RandomForestRegressor
                    from sklearn.preprocessing import LabelEncoder, StandardScaler
                    from sklearn.metrics import mean_squared_error, r2_score
                    from django.utils import timezone
                    
                    # Read the dataset
                    df = pd.read_csv(dataset_file)

                    # --- Feature Engineering to match train_model.py ---
                    # Ensure timestamp is datetime
                    if 'timestamp' in df.columns:
                        df['timestamp'] = pd.to_datetime(df['timestamp'])
                        if 'hour' not in df.columns:
                            df['hour'] = df['timestamp'].dt.hour
                        if 'day_of_week' not in df.columns:
                            df['day_of_week'] = df['timestamp'].dt.dayofweek
                        if 'month' not in df.columns:
                            df['month'] = df['timestamp'].dt.month
                    # Create engineered features if missing
                    if 'temp_humidity_interaction' not in df.columns and 'temperature_celsius' in df.columns and 'humidity_percent' in df.columns:
                        df['temp_humidity_interaction'] = df['temperature_celsius'] * df['humidity_percent']
                    if 'temp_squared' not in df.columns and 'temperature_celsius' in df.columns:
                        df['temp_squared'] = df['temperature_celsius'] ** 2
                    if 'humidity_squared' not in df.columns and 'humidity_percent' in df.columns:
                        df['humidity_squared'] = df['humidity_percent'] ** 2

                    # Drop duplicates and missing values (NaNs)
                    df = df.drop_duplicates().dropna()

                    # Required columns for advanced model
                    required_columns = [
                        'location', 'temperature_celsius', 'humidity_percent', 'rainfall_24h',
                        'days_since_irrigation', 'soil_type', 'hour', 'day_of_week',
                        'month', 'temp_humidity_interaction', 'temp_squared', 'humidity_squared', 'soil_moisture_percent'
                    ]
                    missing_columns = [col for col in required_columns if col not in df.columns]
                    if missing_columns:
                        messages.error(request, f'Missing required columns: {", ".join(missing_columns)}')
                    else:
                        # Prepare features and target
                        feature_cols = [
                            'location', 'temperature_celsius', 'humidity_percent', 'rainfall_24h',
                            'days_since_irrigation', 'soil_type', 'hour', 'day_of_week',
                            'month', 'temp_humidity_interaction', 'temp_squared', 'humidity_squared'
                        ]
                        X = df[feature_cols].copy()
                        y = df['soil_moisture_percent']

                        # Encode categorical variables
                        from sklearn.preprocessing import LabelEncoder, RobustScaler
                        label_encoders = {}
                        for col in ['location', 'soil_type']:
                            le = LabelEncoder()
                            X[col] = le.fit_transform(X[col])
                            label_encoders[col] = le

                        # Scale features
                        scaler = RobustScaler()
                        X_scaled = scaler.fit_transform(X)

                        # Split data
                        from sklearn.model_selection import train_test_split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X_scaled, y, test_size=test_size, random_state=random_state
                        )

                        # Model selection
                        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
                        from sklearn.linear_model import LinearRegression
                        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
                        from sklearn.model_selection import cross_val_score
                        models = {
                            'Random Forest': RandomForestRegressor(n_estimators=200, max_depth=10, random_state=random_state),
                            'Gradient Boosting': GradientBoostingRegressor(n_estimators=200, max_depth=6, random_state=random_state),
                            'Linear Regression': LinearRegression()
                        }
                        best_model = None
                        best_score = -float('inf')
                        best_model_name = ''
                        for name, model in models.items():
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)
                            r2 = r2_score(y_test, y_pred)
                            if r2 > best_score:
                                best_score = r2
                                best_model = model
                                best_model_name = name
                        # Final evaluation
                        y_pred_final = best_model.predict(X_test)
                        mse_final = mean_squared_error(y_test, y_pred_final)
                        rmse_final = np.sqrt(mse_final)
                        r2_final = r2_score(y_test, y_pred_final)
                        mae_final = mean_absolute_error(y_test, y_pred_final)
                        accuracy_percentage = r2_final * 100
                        # Save the model
                        model_path = os.path.join(os.path.dirname(__file__), '..', 'ml', f'{model_name}.pkl')
                        if retrain_existing and os.path.exists(model_path):
                            backup_path = f"{model_path}.backup_{int(timezone.now().timestamp())}"
                            os.rename(model_path, backup_path)
                        model_bundle = {
                            'model': best_model,
                            'scaler': scaler,
                            'label_encoders': label_encoders,
                            'feature_cols': feature_cols,
                            'training_date': timezone.now().isoformat(),
                            'model_name': best_model_name,
                            'metrics': {
                                'mse': mse_final,
                                'rmse': rmse_final,
                                'r2': r2_final,
                                'mae': mae_final,
                                'accuracy_percentage': accuracy_percentage,
                                'n_samples': len(df),
                                'n_train': len(X_train),
                                'n_test': len(X_test)
                            }
                        }
                        with open(model_path, 'wb') as f:
                            pickle.dump(model_bundle, f)
                        training_metrics = {
                            'r2': r2_final,
                            'rmse': rmse_final,
                            'mse': mse_final,
                            'mae': mae_final,
                            'n_samples': len(df),
                            'n_train': len(X_train),
                            'n_test': len(X_test),
                            'model_name': model_name,
                            'r2_percentage': accuracy_percentage,
                            'best_model': best_model_name
                        }
                        messages.success(request, f'✅ Model trained successfully! Best Model: {best_model_name}, R² Score: {r2_final:.4f}, RMSE: {rmse_final:.2f}')
                        logger.info(f'ML model training successful: {model_name} trained with R²={r2_final:.4f}, RMSE={rmse_final:.2f} by user {request.user.username}')
                        # Create notifications for all users about the ML model training
                        from django.contrib.auth.models import User
                        users_with_profiles = User.objects.filter(userprofile__isnull=False)
                        for user in users_with_profiles:
                            try:
                                profile = user.userprofile
                                if profile.role == 'farmer':
                                    notification_message = f"🌾 ML Model Updated: New soil moisture prediction model trained with {accuracy_percentage:.1f}% accuracy."
                                elif profile.role == 'technician':
                                    notification_message = f"🔧 System Update: ML model '{model_name}' retrained with R²={r2_final:.4f}, RMSE={rmse_final:.2f}."
                                elif profile.role == 'admin':
                                    notification_message = f"📊 Model Training Complete: '{model_name}' trained by {request.user.username} with R²={r2_final:.4f}, RMSE={rmse_final:.2f}."
                                else:
                                    notification_message = f"ML Model Updated: '{model_name}' trained with R²={r2_final:.4f}, RMSE={rmse_final:.2f}."
                                Notification.objects.create(
                                    user=user,
                                    message=notification_message
                                )
                            except Exception as e:
                                continue
                        
                except Exception as e:
                    messages.error(request, f'Error training model: {str(e)}')
                    logger.error(f'ML model training error: {str(e)} by user {request.user.username}')
        else:
            # Handle regular CSV upload
            csv_form = SoilMoistureCSVUploadForm(request.POST, request.FILES)
            if csv_form.is_valid():
                csv_file = request.FILES['csv_file']
                try:
                    df = pd.read_csv(csv_file)
                    required_columns = ['location', 'crop_type', 'moisture', 'temperature', 'humidity', 'notes']
                    if not all(col in df.columns for col in required_columns):
                        messages.error(request, 'CSV file is missing required columns.')
                    else:
                        records = []
                        for _, row in df.iterrows():
                            # Get or create location
                            location_name = row['location']
                            location, created = Location.objects.get_or_create(
                                name=location_name,
                                defaults={'latitude': 0.0, 'longitude': 0.0}
                            )
                            
                            # Get or create crop type
                            crop_type_name = row['crop_type']
                            crop_type, created = CropType.objects.get_or_create(
                                name=crop_type_name,
                                defaults={'description': f'Crop type: {crop_type_name}'}
                            )
                            
                            record = SoilMoistureRecord(
                                location=location,
                                crop_type=crop_type,
                                moisture=row['moisture'],
                                temperature=row['temperature'],
                                humidity=row['humidity'],
                                notes=row.get('notes', '')
                            )
                            records.append(record)
                        SoilMoistureRecord.objects.bulk_create(records)
                        messages.success(request, f'Successfully uploaded {len(records)} records.')
                        logger.info(f'CSV upload successful: {len(records)} records uploaded by user {request.user.username}')
                        
                        # Create notifications for all users about the CSV upload
                        from django.contrib.auth.models import User
                        users_with_profiles = User.objects.filter(userprofile__isnull=False)
                        
                        for user in users_with_profiles:
                            try:
                                profile = user.userprofile
                                # Create role-specific notification messages
                                if profile.role == 'farmer':
                                    notification_message = f"🌾 New soil moisture data uploaded: {len(records)} records added to the system."
                                elif profile.role == 'technician':
                                    notification_message = f"🔧 System Update: {len(records)} new soil moisture records uploaded to the database."
                                elif profile.role == 'admin':
                                    notification_message = f"📊 Data Update: {len(records)} soil moisture records uploaded by {request.user.username}."
                                else:
                                    notification_message = f"System Update: {len(records)} new soil moisture records uploaded."
                                
                                Notification.objects.create(
                                    user=user,
                                    message=notification_message
                                )
                            except Exception as e:
                                # Skip users with profile issues
                                continue
                        
                        return redirect('soil_list')
                except Exception as e:
                    messages.error(request, f'Error processing file: {e}')
                    logger.error(f'CSV upload error: {str(e)} by user {request.user.username}')
    
    return render(request, 'soil/csv_upload.html', {
        'form': csv_form,
        'ml_form': ml_form,
        'training_metrics': training_metrics
    })

@api_view(['POST'])
def api_soil_data(request):
    """
    API endpoint for IoT devices to submit soil moisture data
    Enhanced version with better error handling and validation
    """
    try:
        serializer = SoilMoistureRecordSerializer(data=request.data)
        if serializer.is_valid():
            record = serializer.save()
            return Response({
                'success': True,
                'message': 'Soil moisture data recorded successfully',
                'record_id': record.id,
                'data': serializer.data
            }, status=status.HTTP_201_CREATED)
        else:
            return Response({
                'success': False,
                'error': 'Validation failed',
                'details': serializer.errors
            }, status=status.HTTP_400_BAD_REQUEST)
    except Exception as e:
        return Response({
            'success': False,
            'error': 'Internal server error',
            'message': str(e)
        }, status=status.HTTP_500_INTERNAL_SERVER_ERROR)

class SoilMoistureRecordListAPI(generics.ListAPIView):
    queryset = SoilMoistureRecord.objects.select_related('location', 'crop_type').order_by('-recorded_at')
    serializer_class = SoilMoistureRecordSerializer
    filter_backends = [DjangoFilterBackend, filters.OrderingFilter]
    filterset_fields = ['location', 'crop_type']
    ordering_fields = ['recorded_at', 'moisture', 'temperature', 'humidity']

    def get_queryset(self):
        queryset = super().get_queryset()
        start_date = self.request.query_params.get('start_date')
        end_date = self.request.query_params.get('end_date')
        if start_date:
            queryset = queryset.filter(recorded_at__date__gte=start_date)
        if end_date:
            queryset = queryset.filter(recorded_at__date__lte=end_date)
        return queryset

def reports_analytics(request):
    from .models import SoilMoistureRecord, Location, CropType
    from django.db.models.functions import TruncDate
    records = SoilMoistureRecord.objects.select_related('location', 'crop_type').all()
    # Filters
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    location_id = request.GET.get('location')
    crop_type_id = request.GET.get('crop_type')
    if start_date:
        records = records.filter(recorded_at__date__gte=start_date)
    if end_date:
        records = records.filter(recorded_at__date__lte=end_date)
    if location_id:
        records = records.filter(location_id=location_id)
    if crop_type_id:
        records = records.filter(crop_type_id=crop_type_id)
    # Average soil moisture by location (after filters)
    avg_by_location = records.values('location__name').annotate(avg_moisture=Avg('moisture')).order_by('location__name')
    locations = [row['location__name'] for row in avg_by_location]
    avg_moisture = [float(row['avg_moisture']) for row in avg_by_location]
    # Trend: daily average
    trend = records.annotate(day=TruncDate('recorded_at')).values('day').annotate(avg_moisture=Avg('moisture')).order_by('day')
    trend_dates = [row['day'].strftime('%Y-%m-%d') for row in trend]
    trend_moisture = [float(row['avg_moisture']) for row in trend]
    # Min/Max by location
    minmax = records.values('location__name').annotate(min_m=Min('moisture'), max_m=Max('moisture')).order_by('location__name')
    minmax_locations = [row['location__name'] for row in minmax]
    min_moisture = [float(row['min_m']) for row in minmax]
    max_moisture = [float(row['max_m']) for row in minmax]
    # Recent records (last 10, after filters)
    recent_records = records.order_by('-recorded_at')[:10]
    all_locations = Location.objects.all()
    all_crop_types = CropType.objects.all()
    return render(request, 'reports_analytics.html', {
        'locations': locations,
        'avg_moisture': avg_moisture,
        'trend_dates': trend_dates,
        'trend_moisture': trend_moisture,
        'minmax_locations': minmax_locations,
        'min_moisture': min_moisture,
        'max_moisture': max_moisture,
        'recent_records': recent_records,
        'all_locations': all_locations,
        'all_crop_types': all_crop_types,
    })

def export_analytics_csv(request):
    from .models import SoilMoistureRecord, Location, CropType
    records = SoilMoistureRecord.objects.select_related('location', 'crop_type').all()
    # Filters (same as reports_analytics)
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    location_id = request.GET.get('location')
    crop_type_id = request.GET.get('crop_type')
    if start_date:
        records = records.filter(recorded_at__date__gte=start_date)
    if end_date:
        records = records.filter(recorded_at__date__lte=end_date)
    if location_id:
        records = records.filter(location_id=location_id)
    if crop_type_id:
        records = records.filter(crop_type_id=crop_type_id)
    # CSV export
    response = HttpResponse(content_type='text/csv')
    response['Content-Disposition'] = 'attachment; filename="soil_moisture_analytics.csv"'
    writer = csv.writer(response)
    writer.writerow(['Timestamp', 'Location', 'Crop Type', 'Soil Moisture (%)', 'Temperature (°C)', 'Humidity (%)'])
    for rec in records.order_by('-recorded_at'):
        writer.writerow([
            rec.recorded_at.strftime('%Y-%m-%d %H:%M'),
            str(rec.location),
            str(rec.crop_type) if rec.crop_type else '',
            rec.moisture,
            rec.temperature,
            rec.humidity
        ])
    return response

def export_analytics_pdf(request):
    from .models import SoilMoistureRecord, Location, CropType
    from io import BytesIO
    from django.utils import timezone
    from datetime import datetime
    
    try:
        from xhtml2pdf import pisa
    except ImportError:
        return HttpResponse('PDF export requires xhtml2pdf. Please install it.', status=500)
    
    try:
        records = SoilMoistureRecord.objects.select_related('location', 'crop_type').all()
        
        # Filters (same as reports_analytics)
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')
        location_id = request.GET.get('location')
        crop_type_id = request.GET.get('crop_type')
        
        if start_date:
            records = records.filter(recorded_at__date__gte=start_date)
        if end_date:
            records = records.filter(recorded_at__date__lte=end_date)
        if location_id:
            records = records.filter(location_id=location_id)
        if crop_type_id:
            records = records.filter(crop_type_id=crop_type_id)
        
        # Calculate summary statistics for the report
        total_records = records.count()
        avg_moisture = records.aggregate(Avg('moisture'))['moisture__avg'] or 0
        min_moisture = records.aggregate(Min('moisture'))['moisture__min'] or 0
        max_moisture = records.aggregate(Max('moisture'))['moisture__max'] or 0
        
        # Get location and crop type names for display
        location_name = "All Locations"
        crop_type_name = "All Crops"
        
        if location_id:
            try:
                location = Location.objects.get(id=location_id)
                location_name = location.name
            except Location.DoesNotExist:
                pass
                
        if crop_type_id:
            try:
                crop_type = CropType.objects.get(id=crop_type_id)
                crop_type_name = crop_type.name
            except CropType.DoesNotExist:
                pass
        
        # Prepare context for PDF template
        context = {
            'records': records.order_by('-recorded_at'),
            'total_records': total_records,
            'avg_moisture': round(avg_moisture, 2),
            'min_moisture': min_moisture,
            'max_moisture': max_moisture,
            'location_name': location_name,
            'crop_type_name': crop_type_name,
            'start_date': start_date,
            'end_date': end_date,
            'generated_at': timezone.now().strftime('%Y-%m-%d %H:%M:%S'),
            'report_title': 'Soil Moisture Analytics Report'
        }
        
        html = render_to_string('analytics_pdf.html', context)
        
        # Create PDF response
        response = HttpResponse(content_type='application/pdf')
        filename = f"soil_moisture_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        # Generate PDF
        pdf_status = pisa.CreatePDF(BytesIO(html.encode('utf-8')), dest=response)
        
        if pdf_status.err:
            return HttpResponse('Error generating PDF. Please try again.', status=500)
            
        return response
        
    except Exception as e:
        return HttpResponse(f'Error generating PDF: {str(e)}', status=500)

def generate_periodic_report(request, report_type='daily'):
    """
    Generate periodic reports (daily, weekly, monthly)
    """
    from django.utils import timezone
    from datetime import datetime, timedelta
    from .models import SoilMoistureRecord
    
    try:
        # Calculate date range based on report type
        end_date = timezone.now().date()
        
        if report_type == 'daily':
            start_date = end_date - timedelta(days=1)
            report_title = f"Daily Report - {end_date.strftime('%Y-%m-%d')}"
        elif report_type == 'weekly':
            start_date = end_date - timedelta(days=7)
            report_title = f"Weekly Report - {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        elif report_type == 'monthly':
            start_date = end_date - timedelta(days=30)
            report_title = f"Monthly Report - {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        else:
            return HttpResponse('Invalid report type', status=400)
        
        # Get records for the period
        records = SoilMoistureRecord.objects.filter(
            recorded_at__date__gte=start_date,
            recorded_at__date__lte=end_date
        ).select_related('location', 'crop_type')
        
        # Calculate statistics
        total_records = records.count()
        avg_moisture = records.aggregate(Avg('moisture'))['moisture__avg'] or 0
        min_moisture = records.aggregate(Min('moisture'))['moisture__min'] or 0
        max_moisture = records.aggregate(Max('moisture'))['moisture__max'] or 0
        
        # Group by location for location-specific stats
        location_stats = records.values('location__name').annotate(
            avg_moisture=Avg('moisture'),
            min_moisture=Min('moisture'),
            max_moisture=Max('moisture'),
            record_count=Count('id')
        ).order_by('location__name')
        
        context = {
            'records': records.order_by('-recorded_at'),
            'total_records': total_records,
            'avg_moisture': round(avg_moisture, 2),
            'min_moisture': min_moisture,
            'max_moisture': max_moisture,
            'location_stats': location_stats,
            'start_date': start_date,
            'end_date': end_date,
            'generated_at': timezone.now().strftime('%Y-%m-%d %H:%M:%S'),
            'report_title': report_title,
            'report_type': report_type.capitalize()
        }
        
        # Generate PDF
        try:
            from xhtml2pdf import pisa
            from io import BytesIO
            
            html = render_to_string('periodic_report_pdf.html', context)
            response = HttpResponse(content_type='application/pdf')
            filename = f"{report_type}_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            response['Content-Disposition'] = f'attachment; filename="{filename}"'
            
            pdf_status = pisa.CreatePDF(BytesIO(html.encode('utf-8')), dest=response)
            
            if pdf_status.err:
                return HttpResponse('Error generating PDF. Please try again.', status=500)
                
            return response
            
        except ImportError:
            return HttpResponse('PDF export requires xhtml2pdf. Please install it.', status=500)
            
    except Exception as e:
        return HttpResponse(f'Error generating periodic report: {str(e)}', status=500)

def export_prediction_history(request):
    """
    Export historical prediction reports
    """
    from ml.models import PredictionHistory
    from io import BytesIO
    from django.utils import timezone
    from datetime import datetime
    
    try:
        from xhtml2pdf import pisa
    except ImportError:
        return HttpResponse('PDF export requires xhtml2pdf. Please install it.', status=500)
    
    try:
        # Get prediction history
        predictions = PredictionHistory.objects.select_related('user').order_by('-created_at')
        
        # Apply filters if provided
        start_date = request.GET.get('start_date')
        end_date = request.GET.get('end_date')
        
        if start_date:
            predictions = predictions.filter(created_at__date__gte=start_date)
        if end_date:
            predictions = predictions.filter(created_at__date__lte=end_date)
        
        # Calculate statistics
        total_predictions = predictions.count()
        avg_predicted_moisture = predictions.aggregate(Avg('predicted_moisture'))['predicted_moisture__avg'] or 0
        
        # Prepare context for PDF template
        context = {
            'predictions': predictions,
            'total_predictions': total_predictions,
            'avg_predicted_moisture': round(avg_predicted_moisture, 2),
            'start_date': start_date,
            'end_date': end_date,
            'generated_at': timezone.now().strftime('%Y-%m-%d %H:%M:%S'),
            'report_title': 'ML Prediction History Report'
        }
        
        html = render_to_string('prediction_history_pdf.html', context)
        
        # Create PDF response
        response = HttpResponse(content_type='application/pdf')
        filename = f"prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
        response['Content-Disposition'] = f'attachment; filename="{filename}"'
        
        # Generate PDF
        pdf_status = pisa.CreatePDF(BytesIO(html.encode('utf-8')), dest=response)
        
        if pdf_status.err:
            return HttpResponse('Error generating PDF', status=500)
        
        return response
        
    except Exception as e:
        return HttpResponse(f'Error generating PDF: {str(e)}', status=500)

def ml_training_redirect(request):
    """
    Redirect from old ML training URL to the enhanced upload page, with a query param to activate the ML tab
    """
    from django.shortcuts import redirect
    return redirect('/soil/upload-csv/?ml_tab=1')

def iot_api_documentation(request):
    """
    View for IoT API documentation page
    """
    from .models import Location, CropType
    
    locations = Location.objects.all().order_by('name')
    crop_types = CropType.objects.all().order_by('name')
    
    return render(request, 'soil/iot_api_docs.html', {
        'locations': locations,
        'crop_types': crop_types,
    })

def api_documentation(request):
    """
    View for API documentation page
    """
    return render(request, 'soil/api_docs.html', {
        'api_endpoints': {
            'soil_data': {
                'url': '/soil/api/soil-data/',
                'method': 'POST',
                'description': 'Submit soil moisture data',
                'example': {
                    'location': 1,
                    'crop_type': 1,
                    'moisture': 45.5,
                    'temperature': 25.3,
                    'humidity': 60.2,
                    'notes': 'Manual reading'
                }
            },
            'soil_records': {
                'url': '/soil/api/soil-records/',
                'method': 'GET',
                'description': 'Get all soil moisture records',
                'parameters': ['location', 'crop_type', 'start_date', 'end_date']
            },
            'latest_soil_moisture': {
                'url': '/soil/api/latest-soil-moisture/',
                'method': 'GET',
                'description': 'Get the latest soil moisture reading'
            },
            'iot_reading': {
                'url': '/soil/api/iot-reading/',
                'method': 'POST',
                'description': 'Submit IoT device readings',
                'example': {
                    'device_id': 'sensor_001',
                    'location_id': 1,
                    'moisture': 45.5,
                    'temperature': 25.3,
                    'humidity': 60.2,
                    'sensor_type': 'soil_moisture',
                    'crop_type_id': 1,
                    'notes': 'Automated reading'
                }
            },
            'iot_status': {
                'url': '/soil/api/iot-status/',
                'method': 'GET',
                'description': 'Check IoT API status'
            }
        }
    })

def ml_model_training(request):
    """
    View for ML model training with dataset upload
    """
    from .forms import MLModelTrainingForm
    import os
    import pickle
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.metrics import mean_squared_error, r2_score
    import pandas as pd
    from django.utils import timezone
    
    if request.method == 'POST':
        form = MLModelTrainingForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Get uploaded file
                dataset_file = request.FILES['dataset_file']
                model_name = form.cleaned_data['model_name']
                test_size = form.cleaned_data['test_size']
                random_state = form.cleaned_data['random_state']
                retrain_existing = form.cleaned_data['retrain_existing']
                
                # Read the dataset
                df = pd.read_csv(dataset_file)
                
                # Validate required columns
                required_columns = ['location', 'soil_moisture_percent', 'temperature_celsius', 'humidity_percent']
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    messages.error(request, f'Missing required columns: {", ".join(missing_columns)}')
                    return render(request, 'soil/ml_model_training.html', {'form': form})
                
                # Prepare the data
                X = df[['location', 'temperature_celsius', 'humidity_percent']].copy()
                y = df['soil_moisture_percent']
                
                # Encode categorical variables
                label_encoders = {}
                for col in ['location']:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    label_encoders[col] = le
                
                # Split the data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=random_state
                )
                
                # Scale the features
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train the model
                model = RandomForestRegressor(n_estimators=100, random_state=random_state)
                model.fit(X_train_scaled, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)
                
                # Save the model
                model_path = os.path.join(os.path.dirname(__file__), '..', 'ml', f'{model_name}.pkl')
                
                # Create backup if retraining existing model
                if retrain_existing and os.path.exists(model_path):
                    backup_path = f"{model_path}.backup_{int(timezone.now().timestamp())}"
                    os.rename(model_path, backup_path)
                
                # Save model bundle
                model_bundle = {
                    'model': model,
                    'scaler': scaler,
                    'label_encoders': label_encoders,
                    'feature_cols': ['location', 'temperature_celsius', 'humidity_percent'],
                    'training_date': timezone.now().isoformat(),
                    'metrics': {
                        'mse': mse,
                        'rmse': rmse,
                        'r2': r2,
                        'test_size': test_size,
                        'random_state': random_state,
                        'n_samples': len(df),
                        'n_train': len(X_train),
                        'n_test': len(X_test)
                    }
                }
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model_bundle, f)
                
                # Create success message with metrics
                success_message = f"""
                ✅ Model trained successfully!
                
                📊 Training Metrics:
                • R² Score: {r2:.4f}
                • RMSE: {rmse:.2f}
                • MSE: {mse:.2f}
                
                📈 Dataset Info:
                • Total samples: {len(df)}
                • Training samples: {len(X_train)}
                • Test samples: {len(X_test)}
                
                🎯 Model saved as: {model_name}.pkl
                """
                
                messages.success(request, success_message)
                
                # Store metrics in session for display
                request.session['training_metrics'] = {
                    'r2': r2,
                    'rmse': rmse,
                    'mse': mse,
                    'n_samples': len(df),
                    'n_train': len(X_train),
                    'n_test': len(X_test),
                    'model_name': model_name,
                    'r2_percentage': r2 * 100
                }
                
                return redirect('ml_model_training')
                
            except Exception as e:
                messages.error(request, f'Error training model: {str(e)}')
    else:
        form = MLModelTrainingForm()
    
    # Get training metrics from session if available
    training_metrics = request.session.get('training_metrics', None)
    if training_metrics:
        del request.session['training_metrics']  # Clear after displaying
    
    return render(request, 'soil/ml_model_training.html', {
        'form': form,
        'training_metrics': training_metrics
    })
