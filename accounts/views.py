def dashboard_summary(request):
    """Dashboard summary page with real-time soil status and summary widgets."""
    return render(request, 'dashboard_summary.html')
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate
from django.urls import reverse
from .forms import SignUpForm
from django.contrib.auth.decorators import login_required, user_passes_test
from .models import UserProfile
from django.contrib import messages
from django.contrib.auth.models import User
from django.views.decorators.http import require_POST
from django import forms
from django.contrib.auth.forms import PasswordResetForm, UserChangeForm
from django.http import HttpResponseRedirect
from django.http import JsonResponse
from django.contrib.auth.forms import AuthenticationForm
from django.views.decorators.csrf import csrf_exempt
from django.utils.decorators import method_decorator
import json

def custom_login(request):
    """Custom login view that redirects to appropriate dashboard based on user role."""
    if request.user.is_authenticated:
        return redirect_to_dashboard(request.user)
    
    if request.method == 'POST':
        form = AuthenticationForm(request, data=request.POST)
        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')
            user = authenticate(username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect_to_dashboard(user)
            else:
                messages.error(request, 'Invalid username or password.')
        else:
            messages.error(request, 'Invalid username or password.')
    else:
        form = AuthenticationForm()
    
    return render(request, 'registration/login.html', {'form': form})

def redirect_to_dashboard(user):
    """Helper function to redirect user to appropriate dashboard based on role."""
    # Superusers should never be assigned roles or go through custom dashboard logic
    if hasattr(user, 'is_superuser') and user.is_superuser:
        print(f"[DEBUG] Superuser {user.username} detected in redirect_to_dashboard, redirecting to Django admin.")
        return redirect('/admin/')
    
    try:
        # Ensure UserProfile exists and has a role
        profile, created = UserProfile.objects.get_or_create(user=user)
        if not profile.role:
            profile.role = 'farmer'
            profile.save()
        
        role = profile.role
        pending = getattr(profile, 'pending_approval', False)
        
        if role == 'admin':
            if pending:
                return redirect('pending_admin_notice')
            else:
                return redirect('admin_dashboard')
        elif role == 'technician':
            return redirect('technician_dashboard')
        elif role == 'farmer':
            return redirect('farmer_dashboard')
        else:
            messages.error(request, 'Your account does not have a valid role. Please contact support.')
            return redirect('login')
    except Exception as e:
        print(f"[DEBUG] Exception in redirect_to_dashboard: {e}")
        messages.error(request, f'Login failed: {e}. Please contact support.')
        return redirect('login')

def staff_admin_required(view_func):
    return user_passes_test(lambda u: u.is_staff and not u.is_superuser)(view_func)

@staff_admin_required
def delete_notification(request, notification_id):
    from .models import Notification
    notification = get_object_or_404(Notification, id=notification_id, user=request.user)
    notification.delete()
    return HttpResponseRedirect('/accounts/admin/notifications/')

@staff_admin_required
def notification_detail(request, notification_id):
    from .models import Notification
    notification = get_object_or_404(Notification, id=notification_id, user=request.user)
    notification.is_read = True
    notification.save()
    return render(request, 'notification_detail.html', {'notification': notification})


def signup(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect(reverse('login'))  # Redirect to login after registration
    else:
        form = SignUpForm()
    return render(request, "registration/signup.html", {"form": form})


def home(request):
    if request.user.is_authenticated:
        # Robust superuser redirect: nothing else runs for superuser
        if hasattr(request.user, 'is_superuser') and request.user.is_superuser:
            print(f"[DEBUG] Superuser {request.user.username} detected, redirecting to Django admin.")
            return redirect('/admin/')
        return redirect_to_dashboard(request.user)
    user_groups = []
    if request.user.is_authenticated:
        user_groups = list(request.user.groups.values_list('name', flat=True))
    return render(request, "home.html", {"user_groups": user_groups})

@login_required
def technician_dashboard(request):
    user_groups = []
    if request.user.is_authenticated:
        user_groups = list(request.user.groups.values_list('name', flat=True))
    return render(request, "technician_dashboard.html", {"user_groups": user_groups})

@login_required
def farmer_dashboard(request):
    user_groups = []
    if request.user.is_authenticated:
        user_groups = list(request.user.groups.values_list('name', flat=True))
    return render(request, "farmer_dashboard.html", {"user_groups": user_groups})

def is_admin(user):
    try:
        return user.userprofile.role == 'admin' and not user.userprofile.pending_approval
    except Exception:
        return False

@login_required
@user_passes_test(is_admin)
def admin_dashboard(request):
    user_groups = []
    if request.user.is_authenticated:
        user_groups = list(request.user.groups.values_list('name', flat=True))
    return render(request, "admin_dashboard.html", {"user_groups": user_groups})

@login_required
def pending_admin_notice(request):
    messages.warning(request, "Your admin account is pending approval. Please wait for an existing admin to approve your registration.")
    return render(request, "pending_admin.html")

@login_required
@user_passes_test(is_admin)
def approve_admins(request):
    from .models import UserProfile
    pending_admins = UserProfile.objects.filter(role='admin', pending_approval=True)
    if request.method == 'POST':
        approve_ids = request.POST.getlist('approve')
        for profile_id in approve_ids:
            try:
                profile = UserProfile.objects.get(id=profile_id)
                profile.pending_approval = False
                profile.save()
            except UserProfile.DoesNotExist:
                pass
        messages.success(request, 'Selected admins have been approved.')
        return redirect('approve_admins')
    return render(request, 'approve_admins.html', {'pending_admins': pending_admins})

def staff_admin_required(view_func):
    return user_passes_test(lambda u: u.is_staff and not u.is_superuser)(view_func)

@staff_admin_required
def admin_user_management(request):
    users = User.objects.all().select_related('userprofile')
    return render(request, 'admin_user_management.html', {'users': users})

@staff_admin_required
@require_POST
def approve_admin(request, user_id):
    user = get_object_or_404(User, pk=user_id)
    profile = user.userprofile
    profile.pending_approval = False
    profile.save()
    messages.success(request, f"Admin {user.username} has been approved successfully.")
    return redirect(reverse('admin_user_management'))

class AdminEditUserForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'email', 'is_active']
    role = forms.ChoiceField(choices=[('admin', 'Administrator'), ('technician', 'Technician'), ('farmer', 'Farmer')], required=True)

    def __init__(self, *args, **kwargs):
        userprofile = kwargs.pop('userprofile', None)
        super().__init__(*args, **kwargs)
        if userprofile:
            self.fields['role'].initial = userprofile.role
        self.fields['username'].widget.attrs.update({'class': 'form-control'})
        self.fields['email'].widget.attrs.update({'class': 'form-control'})
        self.fields['role'].widget.attrs.update({'class': 'form-select'})
        self.fields['is_active'].widget.attrs.update({'class': 'form-check-input'})

@staff_admin_required
def admin_edit_user(request, user_id):
    user = get_object_or_404(User, pk=user_id)
    profile, created = UserProfile.objects.get_or_create(user=user)
    if request.method == 'POST':
        form = AdminEditUserForm(request.POST, instance=user, userprofile=profile)
        if form.is_valid():
            form.save()
            profile.role = form.cleaned_data['role']
            profile.save()
            return redirect('admin_user_management')
    else:
        form = AdminEditUserForm(instance=user, userprofile=profile, initial={'role': profile.role})
    return render(request, 'admin_edit_user.html', {'form': form, 'user_obj': user})

@staff_admin_required
def admin_delete_user(request, user_id):
    user = get_object_or_404(User, pk=user_id)
    if user == request.user:
        messages.error(request, "You cannot delete your own account.")
        return redirect('admin_user_management')
    if request.method == 'POST':
        user.delete()
        messages.success(request, "User deleted successfully.")
        return redirect('admin_user_management')
    return render(request, 'admin_confirm_delete_user.html', {'user_obj': user})

@staff_admin_required
def admin_reset_password(request, user_id):
    user = get_object_or_404(User, pk=user_id)
    profile, _ = UserProfile.objects.get_or_create(user=user)
    if request.method == 'POST':
        form = PasswordResetForm({'email': user.email})
        if form.is_valid():
            form.save(
                request=request,
                use_https=request.is_secure(),
                email_template_name='registration/password_reset_email.txt',
                html_email_template_name='registration/password_reset_email.html'
            )
            profile.password_reset_requested = False
            profile.save()
            messages.success(request, f"Password reset email sent to {user.email}.")
            return redirect('admin_user_management')
    else:
        form = PasswordResetForm(initial={'email': user.email})
    return render(request, 'admin_reset_password.html', {'form': form, 'user_obj': user})

class AdminProfileForm(forms.ModelForm):
    class Meta:
        model = User
        fields = ['username', 'first_name', 'last_name', 'email']
        widgets = {
            'username': forms.TextInput(attrs={'class': 'form-control'}),
            'first_name': forms.TextInput(attrs={'class': 'form-control'}),
            'last_name': forms.TextInput(attrs={'class': 'form-control'}),
            'email': forms.EmailInput(attrs={'class': 'form-control'}),
        }

@staff_admin_required
def admin_profile_settings(request):
    user = request.user
    if request.method == 'POST':
        form = AdminProfileForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            messages.success(request, 'Profile updated successfully.')
            return redirect('admin_profile_settings')
    else:
        form = AdminProfileForm(instance=user)
    return render(request, 'admin_profile_settings.html', {'form': form})

@staff_admin_required
def admin_notifications(request):
    from .models import UserProfile, Notification
    from django.contrib.auth.models import User
    pending_admins = UserProfile.objects.filter(role='admin', pending_approval=True)
    new_users = User.objects.filter(date_joined__gte=request.user.last_login).exclude(id=request.user.id)
    notifications = []
    # Add ML and other notifications from Notification model
    if request.user.is_authenticated:
        user_notifications = Notification.objects.filter(user=request.user).order_by('-created_at')
        for n in user_notifications:
            notifications.append({
                'type': 'Irrigation Alert',
                'message': n.message,
                'url': f'/accounts/notification/{n.id}/'
            })
    # Add admin-specific notifications
    if pending_admins.exists():
        notifications.append({
            'type': 'Pending Admin Approvals',
            'message': f'There are {pending_admins.count()} admin(s) pending approval.',
            'url': '/accounts/admin/users/'
        })
    if new_users.exists():
        notifications.append({
            'type': 'New User Registrations',
            'message': f'{new_users.count()} new user(s) registered since your last login.',
            'url': '/accounts/admin/users/'
        })
    return render(request, 'admin_notifications.html', {'notifications': notifications})

@login_required
def farmer_notifications(request):
    """Notification view for farmers"""
    from .models import Notification
    notifications = Notification.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'farmer_notifications.html', {'notifications': notifications})

@login_required
def technician_notifications(request):
    """Notification view for technicians"""
    from .models import Notification
    notifications = Notification.objects.filter(user=request.user).order_by('-created_at')
    return render(request, 'technician_notifications.html', {'notifications': notifications})

@login_required
def notification_detail_universal(request, notification_id):
    """Universal notification detail view for all user types"""
    from .models import Notification
    notification = get_object_or_404(Notification, id=notification_id, user=request.user)
    notification.is_read = True
    notification.save()
    return render(request, 'notification_detail_universal.html', {'notification': notification})

@login_required
def delete_notification_universal(request, notification_id):
    """Universal notification deletion for all user types"""
    from .models import Notification
    notification = get_object_or_404(Notification, id=notification_id, user=request.user)
    notification.delete()
    
    # Redirect based on user role
    from .models import UserProfile
    try:
        profile = UserProfile.objects.get(user=request.user)
        if profile.role == 'admin':
            return HttpResponseRedirect('/accounts/admin/notifications/')
        elif profile.role == 'farmer':
            return HttpResponseRedirect('/accounts/farmer/notifications/')
        elif profile.role == 'technician':
            return HttpResponseRedirect('/accounts/technician/notifications/')
    except UserProfile.DoesNotExist:
        pass
    
    # Fallback redirect
    return HttpResponseRedirect('/')

@login_required
def get_notification_count(request):
    """API endpoint to get unread notification count for all users"""
    from .models import Notification
    count = Notification.objects.filter(user=request.user, is_read=False).count()
    return JsonResponse({'count': count})

def find_user_by_username_or_email(identifier):
    try:
        return User.objects.get(username=identifier)
    except User.DoesNotExist:
        try:
            return User.objects.get(email=identifier)
        except User.DoesNotExist:
            return None

@csrf_exempt
def request_password_reset(request):
    if request.method != 'POST':
        return JsonResponse({'error': 'Only POST allowed.'}, status=405)
    try:
        data = json.loads(request.body.decode('utf-8'))
        identifier = data.get('username_or_email')
        user = find_user_by_username_or_email(identifier)
        if not user:
            return JsonResponse({'error': 'User not found.'}, status=404)
        # Set flag on profile
        profile, _ = UserProfile.objects.get_or_create(user=user)
        profile.password_reset_requested = True
        profile.save()
        # Notify all admins
        from .models import Notification
        admin_profiles = UserProfile.objects.filter(role='admin', pending_approval=False)
        for admin_profile in admin_profiles:
            Notification.objects.create(
                user=admin_profile.user,
                message=f'User {user.username} has requested a password reset.'
            )
        return JsonResponse({'message': 'Password reset request sent to admins.'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
