from django import forms
from django.contrib.auth.models import User
from .models import UserProfile
import re

ALL_ROLE_CHOICES = [
    ('admin', 'Administrator'),
    ('technician', 'Technician'),
    ('farmer', 'Farmer'),
]

class SignUpForm(forms.ModelForm):
    password = forms.CharField(widget=forms.PasswordInput)
    role = forms.ChoiceField(choices=ALL_ROLE_CHOICES)

    class Meta:
        model = User
        fields = ['username', 'email', 'password', 'role']
        help_texts = {
            'username': None,
        }

    def clean_username(self):
        username = self.cleaned_data['username']
        if not re.match(r'^[a-z0-9]+$', username):
            raise forms.ValidationError('Username must contain only lowercase letters and digits.')
        return username

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data['password'])
        if commit:
            user.save()
            # Only create UserProfile for non-superusers
            if not user.is_superuser:
                profile, created = UserProfile.objects.get_or_create(user=user)
                profile.role = self.cleaned_data['role']
                # Set pending_approval True for new admin registrations
                if profile.role == 'admin':
                    profile.pending_approval = True
                else:
                    profile.pending_approval = False
                profile.save()
        return user

