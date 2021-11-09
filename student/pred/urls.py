from django.urls import path
from pred import views

urlpatterns = [
    path('', views.indexView, name="home"),
    path('prediction', views.predictionView, name="prediction"),
]
