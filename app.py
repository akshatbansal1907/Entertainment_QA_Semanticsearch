# -*- coding: utf-8 -*-

"""
Experiment 7 - Entertainment Semantic Search + Lightweight Extractive QA

Render 512 MB optimized version:
- Uses TF-IDF + cosine similarity for document retrieval.
- Does not use PyTorch or Transformers.
- Uses lightweight sentence scoring for extractive-style answers.
- Loads only small scikit-learn components.
- Supports documents stored in ./documents/*.txt.
"""

from pathlib import Path
import gc
import os
import re

# Keep numerical libraries lightweight on Render.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOC_DIR = BASE_DIR / "documents"

MAX_DOCUMENT_CHARS = 12000
MAX_CONTEXT_SENTENCES = 5
MIN_SIMILARITY = 0.03

ENTERTAINMENT_TERMS = {
    "movie", "movies", "film", "films", "cinema", "actor", "actors",
    "director", "directors", "genre", "genres", "music", "song", "songs",
    "singer", "instrument", "instruments", "video", "game", "games",
    "gaming", "player", "players", "book", "books", "novel", "novels",
    "reading", "read", "poetry", "poem", "poems", "theater", "theatre",
    "play", "plays", "musical", "stage", "acting", "performance",
    "entertainment", "story", "stories", "character", "characters",
    "audience", "literature", "rpg", "puzzle", "strategy", "thriller",
    "comedy", "drama", "romance", "documentary", "fantasy", "adventure",
    "action", "jazz", "rock", "pop", "classical", "hip-hop", "hiphop"
}


# ---------------------------------------------------------
# Document loading
# ---------------------------------------------------------

def load_documents():
    """
    Load text documents from ./documents.

    A fallback is included for files named doc*.txt in the
    repository root, so deployment does not fail if the files
    were uploaded in the root directory.
    """
    paths = sorted(DOC_DIR.glob("*.txt")) if DOC_DIR.exists() else []

    if not paths:
        paths = sorted(BASE_DIR.glob("doc*.txt"))

    loaded_texts = []
    loaded_names = []

    for path in paths:
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).strip()
        except OSError as exc:
            print(f"Could not read {path.name}: {exc}", flush=True)
            continue

        if text:
            loaded_texts.append(text[:MAX_DOCUMENT_CHARS])
            loaded_names.append(path.name)

    if len(loaded_texts) < 1:
        raise FileNotFoundError(
            "No non-empty TXT documents were found. "
            "Add files inside the documents folder."
        )

    return loaded_texts, loaded_names


documents, document_names = load_documents()

# Fit one small TF-IDF matrix during startup.
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=3500,
    sublinear_tf=True,
    dtype="float32"
)

document_matrix = vectorizer.fit_transform(documents)

# Release temporary objects where possible.
gc.collect()


# ---------------------------------------------------------
# Text processing helpers
# ---------------------------------------------------------

def normalize_words(text):
    return set(
        re.findall(r"[a-zA-Z][a-zA-Z0-9'-]*", (text or "").lower())
    )


def tokenize_words(text):
    return re.findall(
        r"[a-zA-Z][a-zA-Z0-9'-]*",
        (text or "").lower()
    )


def clean_text(text):
    return re.sub(r"\s+", " ", text or "").strip()


def clean_answer(text):
    text = clean_text(text)
    if not text:
        return ""

    return text[0].upper() + text[1:]


def looks_like_entertainment_question(question):
    return bool(normalize_words(question) & ENTERTAINMENT_TERMS)


def split_sentences(text):
    """
    Simple sentence splitter that works well for short TXT files.
    """
    text = clean_text(text)

    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [sentence.strip() for sentence in sentences if sentence.strip()]


# ---------------------------------------------------------
# TF-IDF retrieval
# ---------------------------------------------------------

def semantic_search(question):
    query_matrix = vectorizer.transform([question])
    scores = cosine_similarity(query_matrix, document_matrix)[0]

    best_index = int(scores.argmax())

    return (
        document_names[best_index],
        float(scores[best_index]),
        best_index
    )


# ---------------------------------------------------------
# Lightweight extractive-style answer selection
# ---------------------------------------------------------

def sentence_score(sentence, question, document_score=0.0):
    """
    Score a sentence using:
    - Question-word overlap
    - Important non-stopword overlap
    - Small preference for concise sentences

    This is intentionally lightweight and does not use a QA model.
    """
    question_words = normalize_words(question)
    sentence_words = normalize_words(sentence)

    if not question_words or not sentence_words:
        return 0.0

    overlap = question_words & sentence_words
    overlap_score = len(overlap) / max(len(question_words), 1)

    # Give a little extra weight to words appearing in the query.
    exact_word_score = sum(
        1 for word in tokenize_words(question)
        if word in sentence_words
    ) / max(len(tokenize_words(question)), 1)

    # Avoid selecting extremely long sentences when scores are close.
    length_penalty = min(len(sentence) / 500.0, 1.0) * 0.08

    return (
        (overlap_score * 0.62)
        + (exact_word_score * 0.38)
        - length_penalty
        + (document_score * 0.05)
    )


def extractive_style_answer(question, context):
    """
    Select the most relevant sentence(s) from the selected document.
    """
    sentences = split_sentences(context)

    if not sentences:
        return "", 0.0

    ranked = sorted(
        (
            (
                sentence_score(sentence, question),
                index,
                sentence
            )
            for index, sentence in enumerate(sentences)
        ),
        key=lambda item: item[0],
        reverse=True
    )

    best_score, best_index, best_sentence = ranked[0]

    if best_score <= 0:
        fallback = " ".join(sentences[:2])
        return clean_answer(fallback[:900]), 0.0

    selected = [best_sentence]
    used_length = len(best_sentence)

    # Add up to a few nearby high-scoring sentences when useful.
    for score, index, sentence in ranked[1:]:
        if len(selected) >= MAX_CONTEXT_SENTENCES:
            break

        if score <= best_score * 0.55:
            continue

        if used_length + len(sentence) + 1 > 900:
            continue

        # Prefer sentences near the best sentence.
        if abs(index - best_index) <= 2:
            selected.append(sentence)
            used_length += len(sentence) + 1

    # Preserve document order for readability.
    selected = sorted(
        selected,
        key=lambda sentence: sentences.index(sentence)
    )

    answer = clean_answer(" ".join(selected))
    confidence = max(0.0, min(float(best_score), 1.0))

    return answer[:1200], confidence


# ---------------------------------------------------------
# Main QA workflow
# ---------------------------------------------------------

def hybrid_qa_system(question):
    if not question or not question.strip():
        return (
            "Please enter a question related to entertainment.",
            "N/A",
            "0.0000",
            "N/A",
            "Ask about movies, music, video games, books, or theater."
        )

    question = clean_text(question)

    try:
        source_doc, similarity, best_index = semantic_search(question)

        if (
            similarity < MIN_SIMILARITY
            and not looks_like_entertainment_question(question)
        ):
            return (
                "Sorry, this question is outside the assigned "
                "entertainment topic.",
                "N/A",
                f"{similarity:.4f}",
                "Out of scope",
                "Only entertainment-related questions are accepted."
            )

        context = documents[best_index]

        answer, confidence = extractive_style_answer(
            question,
            context
        )

        if not answer:
            return (
                "The relevant document was found, but no answer "
                "could be extracted.",
                source_doc,
                f"{similarity:.4f}",
                "0.0000",
                "Try a shorter and more specific question."
            )

        return (
            answer,
            source_doc,
            f"{similarity:.4f}",
            f"Lightweight confidence: {confidence:.4f}",
            f"Answer selected from {source_doc}"
        )

    except Exception as exc:
        print(f"Application error: {exc}", flush=True)

        return (
            "The question could not be processed. "
            "Please try a shorter question.",
            "N/A",
            "N/A",
            "QA error",
            str(exc)
        )


# ---------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------

with gr.Blocks(
    title="Entertainment QA & Semantic Search Engine"
) as demo:

    gr.Markdown("# 🎬 Entertainment QA & Semantic Search Engine")

    gr.Markdown(
        "Experiment 7: **TF-IDF Semantic Search + Lightweight "
        "Extractive-Style Question Answering.** "
        "The system selects the most relevant entertainment "
        "document and extracts relevant sentence(s)."
    )

    question_box = gr.Textbox(
        label="Your Question",
        placeholder=(
            "Example: Which movie genres can encourage "
            "deeper thinking?"
        ),
        lines=2
    )

    ask_button = gr.Button(
        "🔎 Find Answer",
        variant="primary"
    )

    similarity_box = gr.Textbox(
        label="Document Similarity Score",
        interactive=False
    )

    source_box = gr.Textbox(
        label="Relevant Source Document",
        interactive=False
    )

    qa_score_box = gr.Textbox(
        label="QA Confidence",
        interactive=False
    )

    answer_box = gr.Textbox(
        label="Answer from the Most Relevant Document",
        lines=5,
        interactive=False
    )

    status_box = gr.Textbox(
        label="System Status",
        lines=2,
        interactive=False
    )

    outputs = [
        answer_box,
        source_box,
        similarity_box,
        qa_score_box,
        status_box
    ]

    ask_button.click(
        fn=hybrid_qa_system,
        inputs=question_box,
        outputs=outputs
    )

    question_box.submit(
        fn=hybrid_qa_system,
        inputs=question_box,
        outputs=outputs
    )

    gr.Markdown(
        "### Assigned topic: Entertainment\n"
        "Documents: Movies, Music, Video Games, Books, Theater"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
