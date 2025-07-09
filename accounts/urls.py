from django.urls import path
from . import views

app_name = 'accounts'
urlpatterns = [
    path('register/', views.signup, name='register'),
    path('profile/', views.profile, name='profile'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),
]