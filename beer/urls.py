from django.conf.urls import url
from django.urls import path, include
from . import views

app_name = 'beer'

urlpatterns = [
    path('', views.index, name='index'),
    path('ver1', views.ver1, name='ver1'),
    path('ver2', views.ver2, name='ver2'),
]
