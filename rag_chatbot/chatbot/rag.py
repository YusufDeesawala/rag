import re
import json
import os

DATA_FILE = "qa_pairs.json"

# -----------------------
# Utility functions
# -----------------------
def extract_keywords(text):
    stopwords = {"is", "a", "the", "and", "what", "how", "do", "i", "of", "to", "in"}
    words = re.findall(r"\w+", text.lower())
    return [w for w in words if w not in stopwords]

def load_db():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            return json.load(f)
    return []

def save_db(qa_pairs):
    with open(DATA_FILE, "w") as f:
        json.dump(qa_pairs, f)

# -----------------------
# Core keyword logic
# -----------------------
def add_pair(question, answer):
    qa_pairs = load_db()
    qa_pairs.append({"question": question, "answer": answer})
    save_db(qa_pairs)

def generate_response(query):
    qa_pairs = load_db()
    query_keywords = set(extract_keywords(query))

    best_score = 0
    best_answer = "Sorry, I don't know the answer."

    for pair in qa_pairs:
        q_keywords = set(extract_keywords(pair["question"]))
        score = len(query_keywords & q_keywords)  # keyword overlap
        if score > best_score:
            best_score = score
            best_answer = pair["answer"]

    return best_answer
