# 🎬 Entertainment QA & Semantic Search Engine

Experiment 7: Semantic Search + Extractive Question Answering.

## Features

- True sentence embeddings with `all-MiniLM-L6-v2`
- Cosine similarity for semantic document retrieval
- Exactly one most relevant entertainment document selected
- Extractive QA with `distilbert-base-uncased-distilled-squad`
- Five entertainment documents, each containing at least two paragraphs
- Similarity score shown in the frontend
- Source document shown in the frontend
- QA confidence shown in the frontend
- Out-of-topic questions receive an appropriate message
- Railway-ready public deployment

## Run locally

```bash
pip install -r requirements.txt
python app.py
```

Then open the Gradio URL printed in the terminal.

## Deploy to Railway from GitHub

1. Create a **public GitHub repository**.
2. Upload all files in this project to the repository.
3. In Railway, create a new project and choose **Deploy from GitHub repo**.
4. Select this repository.
5. Railway will build the Python project from `requirements.txt`.
6. The configured start command is:

```bash
python app.py
```

7. After deployment succeeds, open the service's **Settings → Networking** and generate a public domain.

The app reads Railway's `PORT` environment variable and listens on `0.0.0.0`, so the generated Railway URL can reach the Gradio interface.

## Suggested Git commands

```bash
git init
git add .
git commit -m "Build entertainment semantic search and extractive QA engine"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
git push -u origin main
```

## Example questions

- Which movie genres can encourage deeper thinking?
- What are some musical genres?
- How can video games require decision making?
- How can reading encourage deeper thinking?
- What can theater help audiences interpret?

## Out-of-scope example

`What is the capital of France?`

The system should reject this because the assigned topic is entertainment.
