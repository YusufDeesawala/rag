from django.urls import path
from .views import ChatView, AddPairView

urlpatterns = [
    path("chat/", ChatView.as_view()),
    path("add_pair/", AddPairView.as_view()),
]
