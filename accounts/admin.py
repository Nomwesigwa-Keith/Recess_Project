from django.contrib import admin
from .models import UserProfile

@admin.register(UserProfile)
class UserProfileAdmin(admin.ModelAdmin):
    list_display = ('user', 'role', 'pending_approval')
    list_filter = ('role', 'pending_approval')
    search_fields = ('user__username', 'role')
    list_editable = ('role', 'pending_approval')
