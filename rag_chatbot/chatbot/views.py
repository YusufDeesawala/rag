from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .rag import (
    add_pair, generate_response, get_all_pairs,
    read_history, clear_history
)

class ChatView(APIView):
    """
    POST /api/chat/
    body: { "query": "...", "session_id": "optional-string" }
    """
    def post(self, request):
        query = request.data.get("query")
        session_id = request.data.get("session_id")
        if not query:
            return Response({"error": "query is required"}, status=status.HTTP_400_BAD_REQUEST)
        result = generate_response(query, session_id=session_id)
        return Response(result, status=200)

class AddPairView(APIView):
    """
    POST /api/add_pair/
    body: { "question": "...", "answer": "..." }
    """
    def post(self, request):
        question = request.data.get("question")
        answer = request.data.get("answer")
        if not question or not answer:
            return Response({"error": "question and answer are required"}, status=400)
        add_pair(question, answer)
        return Response({"message": "Pair added successfully"}, status=200)

class ListPairsView(APIView):
    """
    GET /api/pairs/
    """
    def get(self, request):
        return Response({"pairs": get_all_pairs()}, status=200)

class HistoryView(APIView):
    """
    GET /api/history/<session_id>/
    DELETE /api/history/<session_id>/
    """
    def get(self, request, session_id: str):
        return Response({"session_id": session_id, "history": read_history(session_id)}, status=200)

    def delete(self, request, session_id: str):
        clear_history(session_id)
        return Response({"message": "history cleared", "session_id": session_id}, status=200)
