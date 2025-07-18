from django.urls import path
from . import views
from accounts.views import admin_profile_settings

urlpatterns = [
    path('', views.soil_list, name='soil_list'),
    path('add/', views.add_soil_record, name='add_soil_record'),
    path('upload-csv/', views.upload_soil_csv, name='upload_soil_csv'),
    path('ml-training/', views.ml_training_redirect, name='ml_model_training'),

    path('api/soil-data/', views.api_soil_data, name='api_soil_data'),
    path('api/soil-records/', views.SoilMoistureRecordListAPI.as_view(), name='api_soil_records'),
    path('api/latest-soil-moisture/', __import__('soil.api_views').api_views.latest_soil_moisture, name='api_latest_soil_moisture'),
    # IoT Device API endpoints
    path('api/iot-reading/', __import__('soil.api_views').api_views.iot_soil_moisture_reading, name='iot_soil_moisture_reading'),
    path('api/iot-status/', __import__('soil.api_views').api_views.iot_api_status, name='iot_api_status'),
    path('iot-docs/', views.iot_api_documentation, name='iot_api_documentation'),
    path('api-docs/', views.api_documentation, name='api_documentation'),
    path('admin/locations/', views.location_list, name='admin_location_list'),
    path('admin/locations/add/', views.admin_add_location, name='admin_add_location'),
    path('admin/locations/edit/<int:location_id>/', views.admin_edit_location, name='admin_edit_location'),
    path('admin/locations/delete/<int:location_id>/', views.admin_delete_location, name='admin_delete_location'),
    path('admin/crops/', views.crop_type_list, name='admin_crop_type_list'),
    path('admin/crops/add/', views.admin_add_crop_type, name='admin_add_crop_type'),
    path('admin/crops/edit/<int:crop_type_id>/', views.admin_edit_crop_type, name='admin_edit_crop_type'),
    path('admin/crops/delete/<int:crop_type_id>/', views.admin_delete_crop_type, name='admin_delete_crop_type'),
    path('admin/profile/', admin_profile_settings, name='admin_profile_settings'),
]
