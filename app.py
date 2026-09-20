# -*- coding: utf-8 -*-
"""
Experiment 7 - Entertainment Semantic Search + Extractive QA

Requirements from the Experiment 7 PPT:
- Semantic search using TRUE sentence embeddings + cosine similarity.
- At least 5 documents.
- At least 2 paragraphs per document.
- Extractive QA from the single most relevant document.

This version removes the duplicate/repeated notebook sections from the original code.
"""

from pathlib import Path
import re
import os
import numpy as np
import gradio as gr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

# -----------------------------
# 1. Configuration
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
DOC_DIR = BASE_DIR / "documents"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QA_MODEL = "distilbert-base-uncased-distilled-squad"

# A conservative threshold for rejecting questions outside the entertainment collection.
RELEVANCE_THRESHOLD = 0.38

# Extra topic words help prevent a very short unrelated question from being accepted
# merely because every document is about entertainment.
ENTERTAINMENT_TERMS = {
    "movie", "movies", "film", "films", "cinema", "actor", "actors", "director",
    "genre", "genres", "music", "song", "songs", "singer", "instrument",
    "video", "game", "games", "gaming", "player", "players", "book", "books",
    "novel", "novels", "reading", "read", "poetry", "poem", "theater", "theatre",
    "play", "plays", "musical", "stage", "acting", "performance", "entertainment",
    "story", "stories", "character", "characters", "audience", "literature",
    "rpg", "puzzle", "strategy", "thriller", "comedy", "drama", "romance",
    "documentary", "fantasy", "adventure", "action", "jazz", "rock", "pop",
    "classical", "hip-hop"
}

# -----------------------------
# 2. Load documents
# -----------------------------
if not DOC_DIR.exists():
    raise FileNotFoundError(
        f"Document folder not found: {DOC_DIR}\n"
        "Create a 'documents' folder and put the five .txt files inside it."
    )

document_paths = sorted(DOC_DIR.glob("*.txt"))

if len(document_paths) < 5:
    raise RuntimeError(
        f"Experiment 7 requires at least 5 documents, but only {len(document_paths)} were found."
    )

documents = []
document_names = []

for path in document_paths:
    text = path.read_text(encoding="utf-8").strip()
    if text:
        documents.append(text)
        document_names.append(path.name)

if len(documents) < 5:
    raise RuntimeError("At least five non-empty documents are required.")

# -----------------------------
# 3. Load TRUE embedding model
# -----------------------------
print("Loading sentence embedding model...")
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

print("Creating document embeddings...")
document_embeddings = embedding_model.encode(
    documents,
    convert_to_numpy=True,
    normalize_embeddings=True,
    show_progress_bar=True
)

# -----------------------------
# 4. Load extractive QA model
# -----------------------------
print("Loading extractive QA model...")
qa_pipeline = pipeline(
    "question-answering",
    model=QA_MODEL,
    tokenizer=QA_MODEL
)

print("All models loaded successfully.")

# -----------------------------
# 5. Helper functions
# -----------------------------
def normalize_words(text):
    return set(re.findall(r"[a-zA-Z]+", text.lower()))

def looks_like_entertainment_question(question):
    words = normalize_words(question)
    return bool(words & ENTERTAINMENT_TERMS)

def semantic_search(question):
    """Return the single most relevant document and cosine similarity."""
    query_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )
    scores = cosine_similarity(query_embedding, document_embeddings)[0]
    best_index = int(np.argmax(scores))
    return document_names[best_index], float(scores[best_index])

def clean_answer(answer):
    answer = re.sub(r"\s+", " ", answer).strip()
    if not answer:
        return ""
    return answer[0].upper() + answer[1:]

def extractive_answer(question, context):
    """Run extractive QA and return answer + QA confidence."""
    result = qa_pipeline(question=question, context=context)
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

    # First semantic routing step.
    source_doc, similarity = semantic_search(question)

    # Reject clearly unrelated questions.
    # We use BOTH a semantic threshold and an entertainment-topic signal.
    if similarity < RELEVANCE_THRESHOLD and not looks_like_entertainment_question(question):
        return (
            "Sorry, this question is outside the assigned topic. "
            "This system is restricted to entertainment topics such as movies, "
            "music, video games, books, and theater.",
            "N/A",
            f"{similarity:.4f}",
            "Out of scope",
            "Only entertainment-related questions are accepted."
        )

    # If the question has no entertainment vocabulary and the similarity is only modest,
    # treat it as out of scope. This prevents questions such as "What is photosynthesis?"
    # from being answered from whichever entertainment document happens to be closest.
    if similarity < 0.48 and not looks_like_entertainment_question(question):
        return (
            "Sorry, I could not match this question confidently to the assigned "
            "entertainment topic. Please ask about movies, music, video games, books, or theater.",
            "N/A",
            f"{similarity:.4f}",
            "Out of scope",
            "Try an entertainment-focused question."
        )

    context = documents[document_names.index(source_doc)]

    try:
        answer, qa_score = extractive_answer(question, context)

        # A very low QA score means the retrieved document is not giving a reliable
        # extractive answer even though it was the closest document.
        if not answer or qa_score < 0.01:
            return (
                "The most relevant entertainment document was found, but it does not "
                "contain a sufficiently precise answer to this question.",
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
        return (
            "The relevant document was found, but the extractive QA model could not "
            "process the question. Please try again with a shorter, clearer question.",
            source_doc,
            f"{similarity:.4f}",
            "QA error",
            str(exc)
        )

# -----------------------------
# 6. Gradio frontend
# -----------------------------
with gr.Blocks(title="Entertainment QA & Semantic Search Engine") as demo:
    gr.Markdown("# 🎬 Entertainment QA & Semantic Search Engine")
    gr.Markdown(
        "Experiment 7: **Semantic Search + Extractive Question Answering**. "
        "The system selects the single most relevant entertainment document, "
        "then extracts the answer from that document."
    )

    with gr.Row():
        with gr.Column(scale=2):
            question_box = gr.Textbox(
                label="Your Question",
                placeholder="Example: Which movie genres can encourage deeper thinking?",
                lines=2
            )
            ask_button = gr.Button("🔎 Find Answer", variant="primary")

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

    ask_button.click(
        fn=hybrid_qa_system,
        inputs=question_box,
        outputs=[
            answer_box,
            source_box,
            similarity_box,
            qa_score_box,
            status_box
        ]
    )

    question_box.submit(
        fn=hybrid_qa_system,
        inputs=question_box,
        outputs=[
            answer_box,
            source_box,
            similarity_box,
            qa_score_box,
            status_box
        ]
    )

if __name__ == "__main__":
    # Railway provides the PORT environment variable.
    # Gradio must listen on 0.0.0.0 so the public Railway proxy can reach it.
    port = int(os.environ.get("PORT", "7860"))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )
