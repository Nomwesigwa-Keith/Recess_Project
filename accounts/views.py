from django.shortcuts import render, redirect
from django.contrib.auth import login
from .forms import SignUpForm

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


def home(request):
    user_groups = []
    if request.user.is_authenticated:
        user_groups = list(request.user.groups.values_list('name', flat=True))
    return render(request, "home.html", {"user_groups": user_groups})
