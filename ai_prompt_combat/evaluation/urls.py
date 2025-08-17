from django.urls import path
from .views import upload_csv, upload_images

urlpatterns = [
    path('upload_csv/', upload_csv, name='upload_csv'),
    path('upload_images/', upload_images, name='upload_images'),
]