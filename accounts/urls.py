from django.urls import path
from django.contrib.auth.views import LogoutView
from . import views

urlpatterns = [
    path('login/', views.custom_login, name='login'),
    path('logout/', LogoutView.as_view(), name='logout'),
    path('dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('register/', views.signup, name='register'),
    path('technician-dashboard/', views.technician_dashboard, name='technician_dashboard'),
    path('farmer-dashboard/', views.farmer_dashboard, name='farmer_dashboard'),
    path('admin-dashboard/', views.admin_dashboard, name='admin_dashboard'),
    path('pending-admin/', views.pending_admin_notice, name='pending_admin_notice'),
    path('approve-admins/', views.approve_admins, name='approve_admins'),
    path('admin/users/', views.admin_user_management, name='admin_user_management'),
    path('admin/users/approve/<int:user_id>/', views.approve_admin, name='approve_admin'),
    path('admin/users/edit/<int:user_id>/', views.admin_edit_user, name='admin_edit_user'),
    path('admin/users/delete/<int:user_id>/', views.admin_delete_user, name='admin_delete_user'),
    path('admin/users/reset-password/<int:user_id>/', views.admin_reset_password, name='admin_reset_password'),
    path('admin/profile/', views.admin_profile_settings, name='admin_profile_settings'),
    path('admin/notifications/', views.admin_notifications, name='admin_notifications'),
    path('notification/<int:notification_id>/', views.notification_detail, name='notification_detail'),
    path('notification/<int:notification_id>/delete/', views.delete_notification, name='delete_notification'),
    path('request-password-reset/', views.request_password_reset, name='request_password_reset'),
    
    # Universal notification URLs for all user types
    path('farmer/notifications/', views.farmer_notifications, name='farmer_notifications'),
    path('technician/notifications/', views.technician_notifications, name='technician_notifications'),
    path('notification-universal/<int:notification_id>/', views.notification_detail_universal, name='notification_detail_universal'),
    path('notification-universal/<int:notification_id>/delete/', views.delete_notification_universal, name='delete_notification_universal'),
    path('api/notification-count/', views.get_notification_count, name='get_notification_count'),
    path('technician/profile/', views.technician_profile_settings, name='technician_profile_settings'),
    path('farmer/profile/', views.farmer_profile_settings, name='farmer_profile_settings'),
]
