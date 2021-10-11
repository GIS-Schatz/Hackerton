from django.urls import path
from . import views
from .views import *

app_name = 'board_app'

urlpatterns = [
    path('', views.index, name='index'),
    path('post', views.post, name='post'),
    path('post/<int:id>', views.detail, name='detail'),

]
