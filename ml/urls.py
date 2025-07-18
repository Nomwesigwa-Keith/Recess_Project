from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict_soil_moisture, name='predict_soil_moisture'),
    path('predictions/', views.ml_predictions_page, name='ml_predictions_page'),
] 