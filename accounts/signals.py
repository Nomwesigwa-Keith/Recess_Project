from django.db.models.signals import post_save
from django.contrib.auth.models import User
from django.dispatch import receiver
from .models import UserProfile

@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        # Always create a UserProfile with default role 'farmer' if not set
        UserProfile.objects.create(user=instance, role='farmer')

