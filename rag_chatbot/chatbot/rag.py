import google.generativeai as genai
import faiss
import numpy as np
import json
import os

API_KEY = "your_api_key_here"  # put your Gemini key here
genai.configure(api_key=API_KEY)

EMBEDDING_MODEL = "models/embedding-001"
GENERATION_MODEL = "gemini-2.5-flash"

INDEX_FILE = "qa_index.faiss"
DATA_FILE = "qa_pairs.json"


def get_embedding(text):
    result = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=text,
        task_type="retrieval_query"
    )
    return np.array(result['embedding']).astype('float32')


def load_or_init_db():
    if os.path.exists(DATA_FILE) and os.path.exists(INDEX_FILE):
        with open(DATA_FILE, 'r') as f:
            qa_pairs = json.load(f)
        index = faiss.read_index(INDEX_FILE)
    else:
        qa_pairs = []
        d = 768
        index = faiss.IndexFlatL2(d)
    return qa_pairs, index


def save_db(qa_pairs, index):
    with open(DATA_FILE, 'w') as f:
        json.dump(qa_pairs, f)
    faiss.write_index(index, INDEX_FILE)


def add_pair(input_text, response_text):
    qa_pairs, index = load_or_init_db()
    embedding = get_embedding(input_text)
    qa_pairs.append({"input": input_text, "response": response_text})
    index.add(np.array([embedding]))
    save_db(qa_pairs, index)


def generate_response(query):
    qa_pairs, index = load_or_init_db()
    query_embedding = get_embedding(query)

    if index.ntotal == 0:
        model = genai.GenerativeModel(GENERATION_MODEL)
        response = model.generate_content(query)
        return response.text

    D, I = index.search(np.array([query_embedding]), 3)
    context = ""
    for idx in I[0]:
        if 0 <= idx < len(qa_pairs):
            pair = qa_pairs[idx]
            context += f"Q: {pair['input']}\nA: {pair['response']}\n\n"

    prompt = f"""
    Based on the following similar Q&A pairs, answer the user.
    Context:
    {context}
    User Query: {query}
    """
    model = genai.GenerativeModel(GENERATION_MODEL)
    response = model.generate_content(prompt)
    return response.text
