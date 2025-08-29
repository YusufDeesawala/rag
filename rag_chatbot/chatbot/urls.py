from django.urls import path
from .views import ChatView, AddPairView, ListPairsView, HistoryView

urlpatterns = [
    path("chat/", ChatView.as_view()),
    path("add_pair/", AddPairView.as_view()),
    path("pairs/", ListPairsView.as_view()),
    path("history/<str:session_id>/", HistoryView.as_view()),
]
