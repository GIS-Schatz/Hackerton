from django.urls import path

import festival

app_name = 'festival'

urlpatterns = [
    path('', festival.views.fest ),

]