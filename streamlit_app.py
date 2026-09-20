from pathlib import Path
import re

import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Entertainment QA & Semantic Search",
    page_icon="🎬",
    layout="wide"
)


# =========================================================
# MODELS
# =========================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
QA_MODEL = "distilbert-base-uncased-distilled-squad"

RELEVANCE_THRESHOLD = 0.38


# =========================================================
# ENTERTAINMENT KEYWORDS
# Used to reject clearly unrelated questions
# =========================================================

ENTERTAINMENT_TERMS = {
    "movie", "movies", "film", "films", "cinema",
    "actor", "actors", "actress", "director",
    "genre", "genres",

    "music", "song", "songs", "singer",
    "instrument", "band", "bands",

    "video", "game", "games", "gaming",
    "player", "players", "console",

    "book", "books", "novel", "novels",
    "reading", "read", "poetry", "poem",
    "literature",

    "theater", "theatre", "play", "plays",
    "musical", "stage", "acting",
    "performance", "performances",
    "audience",

    "entertainment",
    "story", "stories",
    "character", "characters",

    "rpg", "puzzle", "strategy",
    "thriller", "comedy", "drama",
    "romance", "documentary",
    "fantasy", "adventure", "action",

    "jazz", "rock", "pop",
    "classical", "hip-hop"
}


# =========================================================
# HELPER FUNCTIONS
# =========================================================

def normalize_words(text):
    return set(
        re.findall(r"[a-zA-Z]+", text.lower())
    )


def looks_like_entertainment_question(question):
    words = normalize_words(question)
    return bool(words & ENTERTAINMENT_TERMS)


# =========================================================
# LOAD SENTENCE EMBEDDING MODEL
# =========================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        EMBEDDING_MODEL
    )


# =========================================================
# LOAD EXTRACTIVE QA MODEL
# =========================================================

@st.cache_resource
def load_qa_model():

    return pipeline(
        "question-answering",
        model=QA_MODEL,
        tokenizer=QA_MODEL
    )


# =========================================================
# LOAD DOCUMENTS
# =========================================================

@st.cache_data
def load_documents():

    base_dir = Path(__file__).resolve().parent

    document_files = [
        "doc1_movies.txt",
        "doc2_music.txt",
        "doc3_video_games.txt",
        "doc4_books.txt",
        "doc5_theater.txt"
    ]

    documents = []
    document_names = []

    for filename in document_files:

        file_path = base_dir / filename

        if not file_path.exists():

            raise FileNotFoundError(
                f"Document not found: {filename}"
            )

        text = file_path.read_text(
            encoding="utf-8"
        ).strip()

        documents.append(text)
        document_names.append(filename)

    return documents, document_names


# =========================================================
# CREATE DOCUMENT EMBEDDINGS
# =========================================================

@st.cache_resource
def create_document_embeddings(
    _embedding_model,
    documents
):

    embeddings = _embedding_model.encode(
        documents,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    return embeddings


# =========================================================
# SEMANTIC SEARCH
# =========================================================

def semantic_search(
    question,
    embedding_model,
    document_embeddings,
    document_names
):

    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarity_scores = cosine_similarity(
        question_embedding,
        document_embeddings
    )[0]

    best_index = int(
        np.argmax(similarity_scores)
    )

    best_document = document_names[
        best_index
    ]

    best_score = float(
        similarity_scores[best_index]
    )

    return best_document, best_score


# =========================================================
# CLEAN QA ANSWER
# =========================================================

def clean_answer(answer):

    answer = re.sub(
        r"\s+",
        " ",
        answer
    ).strip()

    if not answer:
        return ""

    return (
        answer[0].upper()
        + answer[1:]
    )


# =========================================================
# EXTRACTIVE QUESTION ANSWERING
# =========================================================

def extractive_answer(
    question,
    context,
    qa_model
):

    result = qa_model(
        question=question,
        context=context
    )

    answer = clean_answer(
        result.get("answer", "")
    )

    confidence = float(
        result.get("score", 0.0)
    )

    return answer, confidence


# =========================================================
# MAIN QA SYSTEM
# =========================================================

def hybrid_qa_system(
    question,
    documents,
    document_names,
    embedding_model,
    document_embeddings,
    qa_model
):

    if not question.strip():

        return {
            "answer":
                "Please enter an entertainment-related question.",
            "source": "N/A",
            "similarity": 0.0,
            "confidence": "N/A",
            "status":
                "Please enter a question."
        }


    question = question.strip()


    # -----------------------------------------------------
    # Semantic Search
    # -----------------------------------------------------

    source_document, similarity = semantic_search(
        question,
        embedding_model,
        document_embeddings,
        document_names
    )


    # -----------------------------------------------------
    # Reject clearly unrelated questions
    # -----------------------------------------------------

    if (
        similarity < RELEVANCE_THRESHOLD
        and
        not looks_like_entertainment_question(question)
    ):

        return {

            "answer":
                "Sorry, this question is outside the "
                "assigned topic. This system is restricted "
                "to entertainment topics such as movies, "
                "music, video games, books, and theater.",

            "source": "N/A",

            "similarity": similarity,

            "confidence": "Out of scope",

            "status":
                "Only entertainment-related questions "
                "are accepted."
        }


    # -----------------------------------------------------
    # Find the actual relevant document
    # -----------------------------------------------------

    document_index = document_names.index(
        source_document
    )

    context = documents[
        document_index
    ]


    # -----------------------------------------------------
    # Extractive QA
    # -----------------------------------------------------

    try:

        answer, confidence = extractive_answer(
            question,
            context,
            qa_model
        )


        if not answer:

            return {

                "answer":
                    "The most relevant document was found, "
                    "but a precise answer could not be extracted.",

                "source":
                    source_document,

                "similarity":
                    similarity,

                "confidence":
                    confidence,

                "status":
                    "Try asking a more specific question."
            }


        return {

            "answer":
                answer,

            "source":
                source_document,

            "similarity":
                similarity,

            "confidence":
                confidence,

            "status":
                f"Answer extracted from {source_document}"
        }


    except Exception as error:

        return {

            "answer":
                "The extractive QA model could not process "
                "this question. Please try a shorter question.",

            "source":
                source_document,

            "similarity":
                similarity,

            "confidence":
                "Error",

            "status":
                str(error)
        }


# =========================================================
# USER INTERFACE
# =========================================================

st.title(
    "🎬 Entertainment QA & Semantic Search"
)

st.subheader(
    "Experiment 7 — Semantic Search + Extractive QA"
)

st.write(
    """
This system uses true sentence embeddings and cosine
similarity to identify the most relevant entertainment
document and then performs extractive question answering
from that document.
"""
)


# =========================================================
# INITIALIZE SYSTEM
# =========================================================

try:

    with st.spinner(
        "Loading entertainment documents..."
    ):

        documents, document_names = (
            load_documents()
        )


    with st.spinner(
        "Loading semantic embedding model..."
    ):

        embedding_model = (
            load_embedding_model()
        )


    with st.spinner(
        "Creating document embeddings..."
    ):

        document_embeddings = (
            create_document_embeddings(
                embedding_model,
                documents
            )
        )


    with st.spinner(
        "Loading extractive QA model..."
    ):

        qa_model = load_qa_model()


except Exception as error:

    st.error(
        "The application could not initialize."
    )

    st.exception(error)

    st.stop()


# =========================================================
# SHOW DOCUMENTS
# =========================================================

with st.expander(
    "📚 Documents used by the system"
):

    for name in document_names:

        st.write(
            f"• {name}"
        )


# =========================================================
# QUESTION INPUT
# =========================================================

question = st.text_area(
    "Ask an entertainment question",
    placeholder=(
        "Example: What are some major movie genres?"
    ),
    height=100
)


# =========================================================
# SEARCH BUTTON
# =========================================================

if st.button(
    "🔎 Find Answer",
    type="primary",
    use_container_width=True
):

    result = hybrid_qa_system(

        question,

        documents,

        document_names,

        embedding_model,

        document_embeddings,

        qa_model
    )


    # -----------------------------------------------------
    # ANSWER
    # -----------------------------------------------------

    st.subheader(
        "Answer from the Most Relevant Document"
    )

    st.write(
        result["answer"]
    )


    # -----------------------------------------------------
    # INFORMATION
    # -----------------------------------------------------

    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Document Similarity Score",
            f"{result['similarity']:.4f}"
        )


    with col2:

        if isinstance(
            result["confidence"],
            str
        ):

            confidence_text = (
                result["confidence"]
            )

        else:

            confidence_text = (
                f"{result['confidence']:.4f}"
            )

        st.metric(
            "QA Confidence",
            confidence_text
        )


    with col3:

        st.metric(
            "Relevant Source Document",
            result["source"]
        )


    # -----------------------------------------------------
    # STATUS
    # -----------------------------------------------------

    st.info(
        result["status"]
    )


# =========================================================
# FOOTER
# =========================================================

st.divider()

st.caption(
    "Assigned Topic: Entertainment • Movies • Music • "
    "Video Games • Books • Theater"
)
