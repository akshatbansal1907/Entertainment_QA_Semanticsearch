# -*- coding: utf-8 -*-
"""
Experiment 7 - Entertainment Semantic Search + Lightweight Extractive QA

512 MB Render-friendly version:
- Uses TF-IDF and cosine similarity for document retrieval.
- Uses sentence-level TF-IDF matching for lightweight answer extraction.
- Does not use PyTorch, Transformers, or Sentence-Transformers.
"""

from pathlib import Path
import os
import re
import gc

# Limit numerical-library threads before importing sklearn.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import gradio as gr
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
DOC_DIR = BASE_DIR / "documents"

RELEVANCE_THRESHOLD = 0.03
MAX_DOCUMENT_CHARS = 12000
MAX_SENTENCES = 80
MAX_ANSWER_CHARS = 650

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


def normalize_words(text):
    return set(re.findall(r"[a-zA-Z][a-zA-Z'-]*", text.lower()))


def looks_like_entertainment_question(question):
    return bool(normalize_words(question) & ENTERTAINMENT_TERMS)


def split_sentences(text):
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    cleaned = []

    for sentence in sentences:
        sentence = sentence.strip(" -•\t")
        if len(sentence) >= 20:
            cleaned.append(sentence)

    return cleaned[:MAX_SENTENCES]


def shorten_answer(text):
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^(answer|response)\s*:\s*", "", text, flags=re.I)

    # Remove repeated leading document headings when they are attached
    # directly to the actual sentence.
    text = re.sub(
        r"^(movies?\s+and\s+genres?|music\s+and\s+genres?)\s+",
        "",
        text,
        flags=re.I
    ).strip()

    if len(text) > MAX_ANSWER_CHARS:
        clipped = text[:MAX_ANSWER_CHARS]
        last_stop = max(
            clipped.rfind("."),
            clipped.rfind(";"),
            clipped.rfind(",")
        )
        if last_stop >= 120:
            text = clipped[:last_stop + 1]
        else:
            text = clipped.rsplit(" ", 1)[0] + "..."

    if text:
        text = text[0].upper() + text[1:]

    return text


def load_documents():
    if not DOC_DIR.exists():
        raise FileNotFoundError(
            f"Document folder not found: {DOC_DIR}. "
            "Create a documents folder containing TXT files."
        )

    paths = sorted(DOC_DIR.glob("*.txt"))
    names = []
    texts = []

    for path in paths:
        try:
            text = path.read_text(
                encoding="utf-8",
                errors="ignore"
            ).strip()
        except OSError:
            continue

        if text:
            names.append(path.name)
            texts.append(text[:MAX_DOCUMENT_CHARS])

    if len(texts) < 5:
        raise RuntimeError(
            f"At least five non-empty TXT documents are required; "
            f"found {len(texts)}."
        )

    return names, texts


document_names, documents = load_documents()

# Document-level retrieval.
document_vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=5000,
    dtype="float32"
)

document_matrix = document_vectorizer.fit_transform(documents)

# Sentence-level data for each document.
document_sentences = [split_sentences(text) for text in documents]

sentence_vectorizers = []
sentence_matrices = []

for sentences in document_sentences:
    if sentences:
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            max_features=2500,
            dtype="float32"
        )
        matrix = vectorizer.fit_transform(sentences)
    else:
        vectorizer = None
        matrix = None

    sentence_vectorizers.append(vectorizer)
    sentence_matrices.append(matrix)

gc.collect()


def semantic_search(question):
    query_matrix = document_vectorizer.transform([question])
    scores = cosine_similarity(query_matrix, document_matrix)[0]
    best_index = int(scores.argmax())

    return (
        best_index,
        document_names[best_index],
        float(scores[best_index])
    )


def sentence_score(question, sentence, vectorizer, matrix):
    question_words = normalize_words(question)
    sentence_words = normalize_words(sentence)

    overlap = len(question_words & sentence_words)
    normalized_overlap = overlap / max(len(question_words), 1)

    query_vector = vectorizer.transform([question])
    similarity = float(cosine_similarity(query_vector, matrix)[0].max())

    # Word overlap helps with short questions such as:
    # "What are popular movie genres?"
    score = (0.70 * similarity) + (0.30 * normalized_overlap)
    return score


def extract_lightweight_answer(question, doc_index):
    sentences = document_sentences[doc_index]
    vectorizer = sentence_vectorizers[doc_index]
    matrix = sentence_matrices[doc_index]

    if not sentences or vectorizer is None or matrix is None:
        return "", 0.0

    question_words = normalize_words(question)
    query_vector = vectorizer.transform([question])
    similarities = cosine_similarity(query_vector, matrix)[0]

    ranked = []

    for index, sentence in enumerate(sentences):
        sentence_words = normalize_words(sentence)
        overlap = len(question_words & sentence_words)
        normalized_overlap = overlap / max(len(question_words), 1)

        score = (
            0.70 * float(similarities[index])
            + 0.30 * normalized_overlap
        )

        ranked.append((score, overlap, index, sentence))

    ranked.sort(key=lambda item: (item[0], item[1]), reverse=True)

    best_score, best_overlap, best_index, best_sentence = ranked[0]

    # Select one sentence by default. A second sentence is included only
    # when it has a similar score and adds useful information.
    selected = [best_sentence]

    if len(ranked) > 1:
        second_score, second_overlap, second_index, second_sentence = ranked[1]

        if (
            second_index != best_index
            and second_score >= best_score * 0.82
            and second_overlap >= 1
            and len(best_sentence) + len(second_sentence) < MAX_ANSWER_CHARS
        ):
            selected.append(second_sentence)

    answer = shorten_answer(" ".join(selected))

    # A small confidence-like value for the lightweight matching method.
    confidence = min(1.0, max(0.0, float(best_score)))
    return answer, confidence


def hybrid_qa_system(question):
    if not question or not question.strip():
        return (
            "Please enter a question related to entertainment.",
            "N/A",
            "0.0000",
            "N/A",
            "Ask about movies, music, video games, books, or theater."
        )

    question = question.strip()

    try:
        doc_index, source_doc, similarity = semantic_search(question)

        if (
            similarity < RELEVANCE_THRESHOLD
            and not looks_like_entertainment_question(question)
        ):
            return (
                "Sorry, this question is outside the assigned entertainment topic.",
                "N/A",
                f"{similarity:.4f}",
                "Lightweight confidence: 0.0000",
                "Only entertainment-related questions are accepted."
            )

        answer, confidence = extract_lightweight_answer(
            question,
            doc_index
        )

        if not answer:
            return (
                "The relevant document was found, but no concise answer "
                "could be extracted.",
                source_doc,
                f"{similarity:.4f}",
                "Lightweight confidence: 0.0000",
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
            "The question could not be processed. Please try a shorter question.",
            "N/A",
            "N/A",
            "QA error",
            str(exc)
        )


with gr.Blocks(
    title="Entertainment QA & Semantic Search Engine"
) as demo:
    gr.Markdown("# 🎬 Entertainment QA & Semantic Search Engine")

    gr.Markdown(
        "Experiment 7: **Semantic Search + Lightweight Extractive "
        "Question Answering.** The system selects the most relevant "
        "entertainment document and returns the most relevant sentence."
    )

    question_box = gr.Textbox(
        label="Your Question",
        placeholder="Example: Which movie genres can encourage deeper thinking?",
        lines=2
    )

    ask_button = gr.Button("🔎 Find Answer", variant="primary")

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
