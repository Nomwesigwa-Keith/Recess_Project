from django.urls import path
from . import views

urlpatterns = [
    path('', views.soil_list, name='soil_list'),
    path('add/', views.add_soil_record, name='add_soil_record'),
]
