from django.urls import path
from . import views

<<<<<<< HEAD
urlpatterns = [
    path('register/', views.signup, name='register'),
]
=======
app_name = 'accounts'
urlpatterns = [
    path('register/', views.signup, name='register'),
    path('profile/', views.profile, name='profile'),
    path('profile/edit/', views.edit_profile, name='edit_profile'),
]
>>>>>>> c5de658ab53168b89c7a47579c76fb46f9eda6e6
