# -*- coding: utf-8 -*-
"""Experiment 7 - Entertainment Semantic Search + Extractive QA.

Render 512 MB version:
- Uses TF-IDF vectors + cosine similarity for semantic-style document retrieval.
- Loads only one Transformer model: the extractive QA model, lazily.
- Avoids SentenceTransformer/PyTorch embedding-model memory usage.
- Uses a short context and limited CPU threads for lower RAM usage.
"""

from pathlib import Path
import gc
import os
import re

# Set these before importing libraries that use BLAS/PyTorch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).resolve().parent
DOC_DIR = BASE_DIR / "documents"
QA_MODEL = os.getenv("QA_MODEL", "deepset/tinyroberta-squad2")
RELEVANCE_THRESHOLD = 0.08
MAX_CONTEXT_CHARS = 3500

ENTERTAINMENT_TERMS = {
    "movie", "movies", "film", "films", "cinema", "actor", "actors",
    "director", "genre", "genres", "music", "song", "songs", "singer",
    "instrument", "video", "game", "games", "gaming", "player", "players",
    "book", "books", "novel", "novels", "reading", "read", "poetry",
    "poem", "theater", "theatre", "play", "plays", "musical", "stage",
    "acting", "performance", "entertainment", "story", "stories",
    "character", "characters", "audience", "literature", "rpg", "puzzle",
    "strategy", "thriller", "comedy", "drama", "romance", "documentary",
    "fantasy", "adventure", "action", "jazz", "rock", "pop", "classical",
    "hip-hop"
}

if not DOC_DIR.exists():
    raise FileNotFoundError(
        f"Document folder not found: {DOC_DIR}. "
        "Create a documents folder containing at least five .txt files."
    )

document_paths = sorted(DOC_DIR.glob("*.txt"))
documents = []
document_names = []

for path in document_paths:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if text:
        documents.append(text)
        document_names.append(path.name)

if len(documents) < 5:
    raise RuntimeError(
        f"At least five non-empty TXT documents are required; found {len(documents)}."
    )

# Lightweight retrieval: no SentenceTransformer and no embedding model.
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000,
    dtype="float32"
)
document_matrix = vectorizer.fit_transform(documents)
_qa_pipeline = None


def get_qa_pipeline():
    global _qa_pipeline
    if _qa_pipeline is None:
        print(f"Loading QA model: {QA_MODEL}", flush=True)
        try:
            import torch
            torch.set_num_threads(1)
            try:
                torch.set_num_interop_threads(1)
            except RuntimeError:
                pass
        except Exception:
            pass

        from transformers import pipeline
        _qa_pipeline = pipeline(
            "question-answering",
            model=QA_MODEL,
            tokenizer=QA_MODEL,
            device=-1
        )
        gc.collect()
        print("QA model loaded.", flush=True)
    return _qa_pipeline


def normalize_words(text):
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def looks_like_entertainment_question(question):
    return bool(normalize_words(question) & ENTERTAINMENT_TERMS)


def semantic_search(question):
    query_matrix = vectorizer.transform([question])
    scores = cosine_similarity(query_matrix, document_matrix)[0]
    best_index = int(scores.argmax())
    return document_names[best_index], float(scores[best_index])


def clean_answer(answer):
    answer = re.sub(r"\s+", " ", answer or "").strip()
    return answer[:1].upper() + answer[1:] if answer else ""


def select_context(text, question):
    """Keep context small to reduce QA inference memory and latency."""
    if len(text) <= MAX_CONTEXT_CHARS:
        return text

    sentences = re.split(r"(?<=[.!?])\s+", text)
    question_words = normalize_words(question)
    ranked = sorted(
        sentences,
        key=lambda sentence: len(normalize_words(sentence) & question_words),
        reverse=True
    )

    selected = []
    total = 0
    for sentence in ranked:
        if total + len(sentence) + 1 > MAX_CONTEXT_CHARS:
            continue
        selected.append(sentence)
        total += len(sentence) + 1
        if total >= MAX_CONTEXT_CHARS * 0.8:
            break

    return " ".join(selected)[:MAX_CONTEXT_CHARS] or text[:MAX_CONTEXT_CHARS]


def extractive_answer(question, context):
    qa = get_qa_pipeline()
    result = qa(
        question=question,
        context=select_context(context, question),
        handle_impossible_answer=True
    )
    return clean_answer(result.get("answer", "")), float(result.get("score", 0.0))


def hybrid_qa_system(question):
    if not question or not question.strip():
        return (
            "Please enter a question related to entertainment.",
            "N/A", "0.0000", "N/A",
            "Ask about movies, music, video games, books, or theater."
        )

    question = question.strip()
    try:
        source_doc, similarity = semantic_search(question)

        if similarity < RELEVANCE_THRESHOLD and not looks_like_entertainment_question(question):
            return (
                "Sorry, this question is outside the assigned entertainment topic.",
                "N/A", f"{similarity:.4f}", "Out of scope",
                "Only entertainment-related questions are accepted."
            )

        context = documents[document_names.index(source_doc)]
        answer, qa_score = extractive_answer(question, context)

        if not answer or qa_score < 0.01:
            return (
                "The relevant document was found, but it does not contain a sufficiently precise answer.",
                source_doc, f"{similarity:.4f}", f"QA confidence: {qa_score:.4f}",
                "Try a shorter and more specific question."
            )

        return (
            answer,
            source_doc,
            f"{similarity:.4f}",
            f"QA confidence: {qa_score:.4f}",
            f"Answer extracted from {source_doc}"
        )

    except Exception as exc:
        print(f"Application error: {exc}", flush=True)
        return (
            "The question could not be processed. Please try a shorter question.",
            "N/A", "N/A", "QA error", str(exc)
        )


with gr.Blocks(title="Entertainment QA & Semantic Search Engine") as demo:
    gr.Markdown("# 🎬 Entertainment QA & Semantic Search Engine")
    gr.Markdown(
        "Experiment 7: **Semantic Search + Extractive Question Answering**. "
        "The system selects the most relevant entertainment document and extracts an answer."
    )

    question_box = gr.Textbox(
        label="Your Question",
        placeholder="Example: Which movie genres can encourage deeper thinking?",
        lines=2
    )
    ask_button = gr.Button("🔎 Find Answer", variant="primary")

    similarity_box = gr.Textbox(label="Document Similarity Score", interactive=False)
    source_box = gr.Textbox(label="Relevant Source Document", interactive=False)
    qa_score_box = gr.Textbox(label="QA Confidence", interactive=False)
    answer_box = gr.Textbox(label="Answer from the Most Relevant Document", lines=5, interactive=False)
    status_box = gr.Textbox(label="System Status", lines=2, interactive=False)

    outputs = [answer_box, source_box, similarity_box, qa_score_box, status_box]
    ask_button.click(hybrid_qa_system, inputs=question_box, outputs=outputs)
    question_box.submit(hybrid_qa_system, inputs=question_box, outputs=outputs)

    gr.Markdown(
        "### Assigned topic: Entertainment\n"
        "Documents: Movies, Music, Video Games, Books, Theater"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(server_name="0.0.0.0", server_port=port, show_error=True)
