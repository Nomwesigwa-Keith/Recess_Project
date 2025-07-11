from django.shortcuts import render, redirect
from .models import SoilMoistureRecord
from .forms import SoilMoistureRecordForm

def soil_list(request):
    records = SoilMoistureRecord.objects.select_related('location', 'crop_type').order_by('-recorded_at')[:20]
    return render(request, 'soil/soil_list.html', {'records': records})

def add_soil_record(request):
    if request.method == 'POST':
        form = SoilMoistureRecordForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('soil_list')
    else:
        form = SoilMoistureRecordForm()
    return render(request, 'soil/add_soil_record.html', {'form': form})
