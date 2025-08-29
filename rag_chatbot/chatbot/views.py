from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .rag import generate_response, add_pair

class ChatView(APIView):
    def post(self, request):
        query = request.data.get("query")
        if not query:
            return Response({"error": "Query is required"}, status=status.HTTP_400_BAD_REQUEST)
        answer = generate_response(query)
        return Response({"query": query, "response": answer})


class AddPairView(APIView):
    def post(self, request):
        question = request.data.get("question")
        answer = request.data.get("answer")
        if not question or not answer:
            return Response({"error": "Both question and answer are required"}, status=status.HTTP_400_BAD_REQUEST)
        add_pair(question, answer)
        return Response({"message": "Pair added successfully!"})
