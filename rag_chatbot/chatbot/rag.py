import os
import re
import json
import uuid
import time
import joblib
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer
from scipy import sparse

# --------- Paths / files ----------
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "qa_pairs.json"            # knowledge base
INDEX_DIR = BASE_DIR / "index"
HISTORY_DIR = BASE_DIR / "histories"
VECTORIZER_FILE = INDEX_DIR / "tfidf_vectorizer.joblib"
MATRIX_FILE = INDEX_DIR / "tfidf_matrix.npz"
QUESTIONS_FILE = INDEX_DIR / "questions.json"     # to align rows of tfidf matrix

INDEX_DIR.mkdir(exist_ok=True)
HISTORY_DIR.mkdir(exist_ok=True)

# --------- Simple stem + stopwords ----------
ps = PorterStemmer()
STOPWORDS = {
    "is","a","the","and","what","how","do","i","of","to","in","on",
    "for","it","that","this","be","an","are","with","as","at","by",
    "from","or","we","you","your","about","me","my","can","could"
}

def extract_keywords(text: str) -> List[str]:
    words = re.findall(r"\w+", (text or "").lower())
    return [ps.stem(w) for w in words if w not in STOPWORDS and len(w) > 2]

def keyword_score(query: str, candidate: str) -> float:
    q = set(extract_keywords(query))
    c = set(extract_keywords(candidate))
    if not q or not c:
        return 0.0
    overlap = len(q & c)
    # Normalize by query keyword count so shorter queries don’t dominate
    return overlap / max(1, len(q))

# --------- Data IO ----------
def load_db() -> List[Dict[str, str]]:
    if DATA_FILE.exists():
        return json.loads(DATA_FILE.read_text(encoding="utf-8"))
    return []

def save_db(pairs: List[Dict[str, str]]) -> None:
    DATA_FILE.write_text(json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8")

def load_questions() -> List[str]:
    if QUESTIONS_FILE.exists():
        return json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))
    return []

def save_questions(questions: List[str]) -> None:
    QUESTIONS_FILE.write_text(json.dumps(questions, ensure_ascii=False, indent=2), encoding="utf-8")

# --------- TF-IDF index ----------
def _build_index(pairs: List[Dict[str, str]]) -> None:
    questions = [p["question"] for p in pairs]
    if not questions:
        # Save empty placeholders
        save_questions([])
        if VECTORIZER_FILE.exists(): VECTORIZER_FILE.unlink()
        if MATRIX_FILE.exists(): MATRIX_FILE.unlink()
        return

    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        stop_words="english"
    )
    X = vectorizer.fit_transform(questions)  # shape: (N, V)
    save_questions(questions)
    joblib.dump(vectorizer, VECTORIZER_FILE)
    sparse.save_npz(MATRIX_FILE, X)

def ensure_index_built() -> None:
    # Build if missing or out of sync
    pairs = load_db()
    questions = load_questions()
    needs_build = False
    if not VECTORIZER_FILE.exists() or not MATRIX_FILE.exists():
        needs_build = True
    else:
        try:
            current_q = [p["question"] for p in pairs]
            if current_q != questions:
                needs_build = True
        except Exception:
            needs_build = True
    if needs_build:
        _build_index(pairs)

def tfidf_score(query: str) -> Tuple[Optional[int], float]:
    """Return (best_index, score) for the query against TF-IDF matrix."""
    ensure_index_built()
    if not VECTORIZER_FILE.exists() or not MATRIX_FILE.exists():
        return None, 0.0

    vectorizer: TfidfVectorizer = joblib.load(VECTORIZER_FILE)
    X = sparse.load_npz(MATRIX_FILE)
    if X.shape[0] == 0:
        return None, 0.0

    qv = vectorizer.transform([query])
    sims = cosine_similarity(qv, X).flatten()
    best_idx = int(np.argmax(sims))
    return best_idx, float(sims[best_idx])

# --------- Hybrid ranking ----------
def hybrid_best_match(query: str, pairs: List[Dict[str, str]]) -> Tuple[Optional[int], float, Dict[str, float]]:
    """
    Combine keyword overlap and TF-IDF similarity.
    Returns: (best_index, combined_score, details)
    """
    if not pairs:
        return None, 0.0, {"kw": 0.0, "tfidf": 0.0}

    # TF-IDF candidate
    tfidf_idx, tfidf_s = tfidf_score(query)
    # Keyword candidate (scan all)
    kw_scores = [keyword_score(query, p["question"]) for p in pairs]
    kw_idx = int(np.argmax(kw_scores)) if kw_scores else None
    kw_s = kw_scores[kw_idx] if kw_idx is not None else 0.0

    # Combine with weights
    alpha = 0.6  # weight for TF-IDF
    beta = 0.4   # weight for keyword
    candidates = []

    if tfidf_idx is not None:
        candidates.append((tfidf_idx, alpha * tfidf_s + beta * keyword_score(query, pairs[tfidf_idx]["question"])))
    if kw_idx is not None:
        candidates.append((kw_idx, alpha * (tfidf_s if kw_idx == tfidf_idx else 0.0) + beta * kw_s))

    if not candidates:
        return None, 0.0, {"kw": kw_s, "tfidf": tfidf_s}

    # pick best combined
    best_idx, best_score = max(candidates, key=lambda t: t[1])
    return best_idx, best_score, {"kw": kw_s, "tfidf": tfidf_s}

# --------- History ----------
def _history_file(session_id: str) -> Path:
    return HISTORY_DIR / f"{session_id}.json"

def append_history(session_id: str, role: str, message: str) -> None:
    hist_path = _history_file(session_id)
    history = []
    if hist_path.exists():
        history = json.loads(hist_path.read_text(encoding="utf-8"))
    history.append({
        "ts": int(time.time()),
        "role": role,
        "message": message
    })
    hist_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

def read_history(session_id: str) -> List[Dict[str, str]]:
    hist_path = _history_file(session_id)
    if hist_path.exists():
        return json.loads(hist_path.read_text(encoding="utf-8"))
    return []

def clear_history(session_id: str) -> None:
    hist_path = _history_file(session_id)
    if hist_path.exists():
        hist_path.unlink()

# --------- Public API used by views ----------
def add_pair(question: str, answer: str) -> None:
    pairs = load_db()
    pairs.append({"question": question, "answer": answer})
    save_db(pairs)
    _build_index(pairs)  # rebuild index on write (simple + safe)

def style_response(answer: str, context_hint: Optional[str] = None) -> str:
    # light templating for a friendlier vibe
    prefaces = [
        "Here’s what I found:",
        "Great question — here's a quick take:",
        "Sure! This should help:",
        "Got it. Summary:",
    ]
    if context_hint:
        return f"{np.random.choice(prefaces)} {answer}\n\n_{context_hint}_"
    return f"{np.random.choice(prefaces)} {answer}"

def generate_response(query: str, session_id: Optional[str] = None) -> Dict[str, str]:
    """
    Returns dict: { 'response': str, 'matched_question': str|None, 'scores': {'kw': float, 'tfidf': float, 'combined': float}, 'session_id': str }
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    pairs = load_db()
    best_idx, combined, details = hybrid_best_match(query, pairs)

    append_history(session_id, "user", query)

    THRESHOLD = 0.12  # tune as needed
    if best_idx is None or combined < THRESHOLD:
        msg = "Hmm, I’m not fully sure yet. Could you rephrase or add more detail?"
        append_history(session_id, "bot", msg)
        return {
            "response": msg,
            "matched_question": None,
            "scores": {"combined": combined, **details},
            "session_id": session_id
        }

    best_pair = pairs[best_idx]
    ctx = f"Matched saved Q: \"{best_pair['question']}\""
    final = style_response(best_pair["answer"], context_hint=ctx)
    append_history(session_id, "bot", final)

    return {
        "response": final,
        "matched_question": best_pair["question"],
        "scores": {"combined": combined, **details},
        "session_id": session_id
    }

def get_all_pairs() -> List[Dict[str, str]]:
    return load_db()
