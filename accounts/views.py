from django.shortcuts import render, redirect
from django.contrib.auth import login
from django.contrib.auth.decorators import login_required
from .forms import SignUpForm, UserProfileForm, ProfileForm
from .models import UserProfile

def signup(request):
    if request.method == "POST":
        form = SignUpForm(request.POST)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect("/")
    else:
      form = SignUpForm()
    return render(request, "registration/signup.html", {"form": form})

@login_required
def profile(request):
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    return render(request, 'accounts/profile.html', {'profile': profile})

@login_required
def edit_profile(request):
    profile, created = UserProfile.objects.get_or_create(user=request.user)
    if request.method == 'POST':
        user_form = UserProfileForm(request.POST, instance=request.user)
        profile_form = ProfileForm(request.POST, instance=profile)
        if user_form.is_valid() and profile_form.is_valid():
            user_form.save()
            profile_form.save()
            return redirect('accounts:profile')
    else:
        user_form = UserProfileForm(instance=request.user)
        profile_form = ProfileForm(instance=profile)
    return render(request, 'accounts/edit_profile.html', {
        'user_form': user_form, 'profile_form': profile_form
    })

def home(request):
    user_groups = []
    if request.user.is_authenticated:
        user_groups = list(request.user.groups.values_list('name', flat=True))
    return render(request, "home.html", {"user_groups": user_groups})
