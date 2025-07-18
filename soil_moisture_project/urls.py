"""
URL configuration for soil_moisture_project project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""

from django.contrib import admin
from django.urls import path, include

from accounts import views as accounts_views
from soil import views as soil_views

urlpatterns = [
    path('', accounts_views.home, name='home'),
    path('dashboard/', accounts_views.admin_dashboard, name='admin_dashboard'),
    path('admin/', admin.site.urls),
    path('accounts/', include('accounts.urls')),
    path('accounts/', include('django.contrib.auth.urls')),  # <-- Added for password change and auth views
    path('soil/', include('soil.urls')),
    path('ml/', include('ml.urls')),
    # Direct analytics routes
    path('reports/analytics/', soil_views.reports_analytics, name='reports_analytics'),
    path('reports/analytics/export/csv/', soil_views.export_analytics_csv, name='export_analytics_csv'),
    path('reports/analytics/export/pdf/', soil_views.export_analytics_pdf, name='export_analytics_pdf'),
    # Periodic reports
    path('reports/periodic/daily/', soil_views.generate_periodic_report, {'report_type': 'daily'}, name='daily_report'),
    path('reports/periodic/weekly/', soil_views.generate_periodic_report, {'report_type': 'weekly'}, name='weekly_report'),
    path('reports/periodic/monthly/', soil_views.generate_periodic_report, {'report_type': 'monthly'}, name='monthly_report'),
    # Prediction history export
    path('reports/predictions/export/', soil_views.export_prediction_history, name='export_prediction_history'),
]
