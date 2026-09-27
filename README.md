# RAG Chatbot — Gemini + Flask + ChromaDB

This project is a simple PDF-based Retrieval-Augmented Generation (RAG) chatbot.

## Project structure

rag_chatbot/
├── app.py
├── .env
├── requirements.txt
├── documents/
├── data/
│   └── chroma/
└── templates/
    └── index.html

## Setup

1. Create/activate a Python virtual environment.
2. Install packages:
   pip install -r requirements.txt
3. Open `.env` and replace the placeholder with your Gemini API key.
4. Put one or more `.pdf` files inside `documents/`.
5. Start the Flask app:
   python app.py
6. Open the local Flask URL shown in the terminal.
7. Click "Rebuild Knowledge Base" after adding or changing PDFs.
8. Ask questions about the PDF content.

## RAG flow

PDF → Text extraction → Chunking → Gemini embeddings → ChromaDB
→ User query embedding → Similar chunks retrieved → Gemini → Answer

The chatbot is instructed not to invent information outside the retrieved context.

## Notes

- The API key is stored in `.env` and should never be uploaded to GitHub.
- `.gitignore` excludes `.env`, the local ChromaDB data, Python cache, and virtual environments.
