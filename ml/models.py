from django.db import models
from django.contrib.auth.models import User

# Create your models here.

class PredictionHistory(models.Model):
    user = models.ForeignKey(User, on_delete=models.SET_NULL, null=True, blank=True)
    location = models.CharField(max_length=100)
    temperature_celsius = models.FloatField()
    humidity_percent = models.FloatField()
    predicted_moisture = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user} - {self.location} - {self.predicted_moisture:.2f}% at {self.created_at.strftime('%Y-%m-%d %H:%M')}"
