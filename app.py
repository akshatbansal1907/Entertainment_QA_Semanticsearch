# -*- coding: utf-8 -*-
"""
Experiment 7 - Entertainment Semantic Search + Extractive QA

Render-friendly version:
- Uses true sentence embeddings + cosine similarity.
- Requires at least five non-empty TXT documents.
- Uses lazy loading for the QA model to reduce startup memory.
- Uses smaller models by default to reduce RAM usage.
- Keeps the Gradio interface and all original output fields.
"""

from pathlib import Path
from functools import lru_cache
import gc
import os
import re

import numpy as np
import gradio as gr
from sklearn.metrics.pairwise import cosine_similarity


# -------------------------------------------------
# 1. Configuration
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DOC_DIR = BASE_DIR / "documents"

# Smaller models help on Render's limited-memory instances.
# They can be overridden with environment variables in Render.
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-MiniLM-L3-v2"
)

QA_MODEL = os.getenv(
    "QA_MODEL",
    "deepset/tinyroberta-squad2"
)

RELEVANCE_THRESHOLD = 0.38

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


# -------------------------------------------------
# 2. Load documents
# -------------------------------------------------

if not DOC_DIR.exists():
    raise FileNotFoundError(
        f"Document folder not found: {DOC_DIR}\n"
        "Create a 'documents' folder and place at least five .txt files inside it."
    )

document_paths = sorted(DOC_DIR.glob("*.txt"))

if len(document_paths) < 5:
    raise RuntimeError(
        f"Experiment 7 requires at least 5 documents, "
        f"but only {len(document_paths)} were found."
    )

documents = []
document_names = []

for path in document_paths:
    text = path.read_text(encoding="utf-8", errors="ignore").strip()

    if text:
        documents.append(text)
        document_names.append(path.name)

if len(documents) < 5:
    raise RuntimeError("At least five non-empty TXT documents are required.")


# -------------------------------------------------
# 3. Lazy model loading
# -------------------------------------------------

_embedding_model = None
_qa_pipeline = None


def get_embedding_model():
    """
    Load the embedding model only once.
    The model remains available for future search requests.
    """
    global _embedding_model

    if _embedding_model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")

        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL,
            device="cpu"
        )

        # Reduce unnecessary CPU thread overhead.
        try:
            _embedding_model.eval()
        except AttributeError:
            pass

        print("Embedding model loaded.")

    return _embedding_model


def get_qa_pipeline():
    """
    Load the QA model only when the first question is submitted.
    This reduces startup memory usage and startup time.
    """
    global _qa_pipeline

    if _qa_pipeline is None:
        print(f"Loading QA model: {QA_MODEL}")

        from transformers import pipeline

        _qa_pipeline = pipeline(
            task="question-answering",
            model=QA_MODEL,
            tokenizer=QA_MODEL,
            device=-1
        )

        print("QA model loaded.")

    return _qa_pipeline


# -------------------------------------------------
# 4. Document embeddings
# -------------------------------------------------

print("Creating document embeddings...")

embedding_model = get_embedding_model()

document_embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=False,
    batch_size=2
)

document_embeddings = np.asarray(
    document_embeddings,
    dtype=np.float32
)

# Release temporary memory after embedding creation.
gc.collect()

print("Document embeddings created successfully.")


# -------------------------------------------------
# 5. Helper functions
# -------------------------------------------------

def normalize_words(text):
    return set(re.findall(r"[a-zA-Z]+", text.lower()))


def looks_like_entertainment_question(question):
    words = normalize_words(question)
    return bool(words & ENTERTAINMENT_TERMS)


def semantic_search(question):
    """
    Return the single most relevant document and cosine similarity.
    """
    model = get_embedding_model()

    query_embedding = model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype=np.float32
    )

    scores = cosine_similarity(
        query_embedding,
        document_embeddings
    )[0]

    best_index = int(np.argmax(scores))

    return (
        document_names[best_index],
        float(scores[best_index])
    )


def clean_answer(answer):
    answer = re.sub(r"\s+", " ", answer).strip()

    if not answer:
        return ""

    return answer[0].upper() + answer[1:]


def extractive_answer(question, context):
    """
    Run extractive QA and return answer plus QA confidence.
    The QA pipeline is loaded only when required.
    """
    qa = get_qa_pipeline()

    result = qa(
        question=question,
        context=context
    )

    answer = clean_answer(result.get("answer", ""))
    score = float(result.get("score", 0.0))

    return answer, score


def hybrid_qa_system(question):
    if not question or not question.strip():
        return (
            "Please enter a question related to entertainment.",
            "N/A",
            "0.0000",
            "N/A",
            "Enter a question about movies, music, video games, books, or theater."
        )

    question = question.strip()

    try:
        # Step 1: Semantic routing.
        source_doc, similarity = semantic_search(question)

        # Step 2: Reject clearly unrelated questions.
        if (
            similarity < RELEVANCE_THRESHOLD
            and not looks_like_entertainment_question(question)
        ):
            return (
                "Sorry, this question is outside the assigned topic. "
                "This system is restricted to entertainment topics such as movies, "
                "music, video games, books, and theater.",
                "N/A",
                f"{similarity:.4f}",
                "Out of scope",
                "Only entertainment-related questions are accepted."
            )

        # Step 3: Reject questions with weak semantic similarity.
        if (
            similarity < 0.48
            and not looks_like_entertainment_question(question)
        ):
            return (
                "Sorry, I could not match this question confidently to the assigned "
                "entertainment topic. Please ask about movies, music, video games, "
                "books, or theater.",
                "N/A",
                f"{similarity:.4f}",
                "Out of scope",
                "Try an entertainment-focused question."
            )

        context = documents[document_names.index(source_doc)]

        # Step 4: Extractive QA from the selected document.
        answer, qa_score = extractive_answer(
            question,
            context
        )

        if not answer or qa_score < 0.01:
            return (
                "The most relevant entertainment document was found, but it does "
                "not contain a sufficiently precise answer to this question.",
                source_doc,
                f"{similarity:.4f}",
                f"QA confidence: {qa_score:.4f}",
                "Try asking a more specific question about the retrieved topic."
            )

        return (
            answer,
            source_doc,
            f"{similarity:.4f}",
            f"QA confidence: {qa_score:.4f}",
            f"Answer extracted from {source_doc}"
        )

    except Exception as exc:
        print(f"Application error: {exc}")

        return (
            "The relevant document was found, but the extractive QA model could "
            "not process the question. Please try again with a shorter, clearer question.",
            "N/A",
            "N/A",
            "QA error",
            str(exc)
        )


# -------------------------------------------------
# 6. Gradio frontend
# -------------------------------------------------

with gr.Blocks(
    title="Entertainment QA & Semantic Search Engine"
) as demo:

    gr.Markdown(
        "# 🎬 Entertainment QA & Semantic Search Engine"
    )

    gr.Markdown(
        "Experiment 7: **Semantic Search + Extractive Question Answering**. "
        "The system selects the single most relevant entertainment document, "
        "then extracts the answer from that document."
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Your Question",
                placeholder=(
                    "Example: Which movie genres can encourage deeper thinking?"
                ),
                lines=2
            )

            ask_button = gr.Button(
                "🔎 Find Answer",
                variant="primary"
            )

        with gr.Column(scale=1):
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

    gr.Markdown(
        "### Assigned topic: Entertainment\n"
        "Documents: Movies, Music, Video Games, Books, Theater"
    )

    output_components = [
        answer_box,
        source_box,
        similarity_box,
        qa_score_box,
        status_box
    ]

    ask_button.click(
        fn=hybrid_qa_system,
        inputs=question_box,
        outputs=output_components
    )

    question_box.submit(
        fn=hybrid_qa_system,
        inputs=question_box,
        outputs=output_components
    )


# -------------------------------------------------
# 7. Start the Gradio application
# -------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7860"))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
