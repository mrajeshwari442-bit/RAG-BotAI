import os
import time
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pypdf import PdfReader

import chromadb
from google import genai


# =========================================================
# LOAD ENVIRONMENT
# =========================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GENERATION_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.8-flash"
)

EMBEDDING_MODEL = os.getenv(
    "GEMINI_EMBEDDING_MODEL",
    "gemini-embedding-001"
)

if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing in the .env file."
    )


# =========================================================
# FLASK
# =========================================================

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = BASE_DIR / "documents"
CHROMA_DIR = BASE_DIR / "data" / "chroma"

DOCUMENTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

CHROMA_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# =========================================================
# GEMINI CLIENT
# =========================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# =========================================================
# CHROMADB
# =========================================================

chroma_client = chromadb.PersistentClient(
    path=str(CHROMA_DIR)
)


def get_collection():

    return chroma_client.get_or_create_collection(
        name="rag_documents"
    )


collection = get_collection()


# =========================================================
# TEXT CHUNKING
# =========================================================

def chunk_text(
    text,
    chunk_size=1200,
    overlap=200
):

    text = text.strip()

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# =========================================================
# EXTRACT PDF TEXT
# =========================================================

def extract_pdf_text(pdf_path):

    reader = PdfReader(
        str(pdf_path)
    )

    pages_text = []

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        try:

            page_text = page.extract_text()

            if page_text:
                pages_text.append(
                    f"[Page {page_number}]\n"
                    f"{page_text}"
                )

        except Exception as error:

            print(
                f"PDF page extraction error "
                f"on page {page_number}: {error}"
            )

    return "\n\n".join(
        pages_text
    )


# =========================================================
# GEMINI EMBEDDING
# =========================================================

def create_embedding(text):

    response = client.models.embed_content(
        model=EMBEDDING_MODEL,
        contents=text
    )

    if not response or not response.embeddings:

        raise RuntimeError(
            "Gemini embedding response was empty."
        )

    return response.embeddings[0].values


# =========================================================
# CLEAR CHROMA COLLECTION
# =========================================================

def clear_collection():

    global collection

    try:

        chroma_client.delete_collection(
            name="rag_documents"
        )

    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(
        name="rag_documents"
    )

    print(
        "ChromaDB collection cleared."
    )


# =========================================================
# DELETE ALL PDF FILES
# =========================================================

def delete_all_pdfs():

    for pdf_file in DOCUMENTS_DIR.glob("*.pdf"):

        try:

            pdf_file.unlink()

            print(
                f"Deleted old PDF: "
                f"{pdf_file.name}"
            )

        except Exception as error:

            print(
                f"Could not delete "
                f"{pdf_file.name}: {error}"
            )


# =========================================================
# REBUILD RAG INDEX
# =========================================================

def rebuild_index():

    global collection

    pdf_files = list(
        DOCUMENTS_DIR.glob("*.pdf")
    )

    if not pdf_files:

        print(
            "No PDF found."
        )

        return 0

    total_chunks = 0

    for pdf_path in pdf_files:

        print(
            f"Reading PDF: "
            f"{pdf_path.name}"
        )

        text = extract_pdf_text(
            pdf_path
        )

        if not text.strip():

            print(
                "No extractable text found "
                f"in {pdf_path.name}"
            )

            continue

        chunks = chunk_text(
            text,
            chunk_size=1200,
            overlap=200
        )

        print(
            f"Created {len(chunks)} chunks."
        )

        for index, chunk in enumerate(chunks):

            print(
                f"Creating embedding "
                f"{index + 1}/{len(chunks)}..."
            )

            embedding = create_embedding(
                chunk
            )

            chunk_id = (
                f"{pdf_path.stem}_chunk_{index}"
            )

            collection.add(
                ids=[chunk_id],
                embeddings=[embedding],
                documents=[chunk],
                metadatas=[
                    {
                        "source":
                            pdf_path.name,

                        "chunk":
                            index
                    }
                ]
            )

            total_chunks += 1

    print(
        f"RAG indexing completed: "
        f"{total_chunks} chunks"
    )

    return total_chunks


# =========================================================
# RETRIEVE RELEVANT CONTEXT
# =========================================================

def retrieve_context(
    question,
    top_k=5
):

    if collection.count() == 0:

        return ""

    query_embedding = create_embedding(
        question
    )

    results = collection.query(
        query_embeddings=[
            query_embedding
        ],
        n_results=min(
            top_k,
            collection.count()
        )
    )

    documents = results.get(
        "documents",
        []
    )

    if not documents:

        return ""

    retrieved_documents = documents[0]

    if not retrieved_documents:

        return ""

    context_parts = []

    for index, document in enumerate(
        retrieved_documents,
        start=1
    ):

        context_parts.append(
            f"--- Context {index} ---\n"
            f"{document}"
        )

    return "\n\n".join(
        context_parts
    )


# =========================================================
# GENERATE GEMINI ANSWER
# =========================================================

def generate_answer(
    question,
    context_text
):

    prompt = f"""
You are a Retrieval-Augmented Generation (RAG)
document assistant.

Your job is to answer the user's question ONLY
using the retrieved context provided below.

Retrieved Context:
{context_text}

User Question:
{question}

Rules:

1. Use only the retrieved context.
2. Do not use outside knowledge.
3. Do not invent or guess information.
4. If the answer is not available in the context,
say exactly:

"I could not find this information in the uploaded document."

5. Answer in the same language/style as the user
whenever possible.
6. English, Tamil and Thanglish are supported.
7. Keep the answer simple, clear and useful.
8. Do not use LaTeX.
9. Simple Markdown is allowed.
10. Use headings, bullet points or numbered lists
when they make the answer easier to understand.
"""

    last_error = None

    # -----------------------------------------------------
    # RETRY GEMINI TEMPORARY ERRORS
    # -----------------------------------------------------

    for attempt in range(5):

        try:

            print(
                f"Gemini generation attempt "
                f"{attempt + 1}/5..."
            )

            response = client.models.generate_content(
                model=GENERATION_MODEL,
                contents=prompt
            )

            if response and response.text:

                answer = response.text.strip()

                print(
                    "Gemini answer generated successfully."
                )

                return answer

            return (
                "I could not generate an answer."
            )

        except Exception as error:

            last_error = error

            error_text = str(
                error
            )

            print(
                f"Gemini error: {error_text}"
            )

            error_lower = (
                error_text.lower()
            )

            is_temporary = (
                "503" in error_text
                or "unavailable" in error_lower
                or "high demand" in error_lower
                or "temporarily" in error_lower
                or "429" in error_text
                or "rate limit" in error_lower
                or "resource exhausted" in error_lower
            )

            if (
                is_temporary
                and attempt < 4
            ):

                wait_time = 2 ** attempt

                print(
                    f"Temporary Gemini error. "
                    f"Retrying in {wait_time} seconds..."
                )

                time.sleep(
                    wait_time
                )

                continue

            break

    # -----------------------------------------------------
    # FINAL ERROR
    # -----------------------------------------------------

    if last_error:

        error_text = str(
            last_error
        ).lower()

        if (
            "503" in error_text
            or "unavailable" in error_text
            or "high demand" in error_text
            or "temporarily" in error_text
        ):

            return (
                "Gemini is temporarily busy right now. "
                "Please try your question again in a few seconds."
            )

        if (
            "429" in error_text
            or "rate limit" in error_text
            or "resource exhausted" in error_text
        ):

            return (
                "Gemini API quota or rate limit was reached. "
                "Please wait a little and try again."
            )

        print(
            f"Final Gemini error: "
            f"{last_error}"
        )

        return (
            "Sorry, I could not generate the answer "
            "right now."
        )

    return (
        "Sorry, I could not generate the answer."
    )


# =========================================================
# MAIN RAG ANSWER
# =========================================================

def answer_question(
    question
):

    if collection.count() == 0:

        return (
            "Please upload a PDF document first."
        )

    try:

        context = retrieve_context(
            question,
            top_k=5
        )

    except Exception as error:

        print(
            f"Retrieval error: {error}"
        )

        return (
            "I could not retrieve information "
            "from the uploaded document."
        )

    if not context:

        return (
            "I could not find relevant information "
            "in the uploaded document."
        )

    return generate_answer(
        question,
        context
    )


# =========================================================
# HOME
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# UPLOAD PDF
# =========================================================

@app.route(
    "/api/upload",
    methods=["POST"]
)
def upload_pdf():

    try:

        if "file" not in request.files:

            return jsonify(
                {
                    "success": False,
                    "error":
                        "No file selected."
                }
            ), 400

        file = request.files[
            "file"
        ]

        if not file.filename:

            return jsonify(
                {
                    "success": False,
                    "error":
                        "No file selected."
                }
            ), 400

        if not file.filename.lower().endswith(
            ".pdf"
        ):

            return jsonify(
                {
                    "success": False,
                    "error":
                        "Only PDF files are supported."
                }
            ), 400

        # -------------------------------------------------
        # DELETE PREVIOUS DOCUMENT
        # -------------------------------------------------

        print(
            "Deleting previous document..."
        )

        delete_all_pdfs()

        clear_collection()

        # -------------------------------------------------
        # SAVE NEW PDF
        # -------------------------------------------------

        safe_filename = Path(
            file.filename
        ).name

        pdf_path = (
            DOCUMENTS_DIR /
            safe_filename
        )

        file.save(
            str(pdf_path)
        )

        print(
            f"Uploaded: "
            f"{safe_filename}"
        )

        # -------------------------------------------------
        # BUILD RAG AUTOMATICALLY
        # -------------------------------------------------

        print(
            "Building RAG index..."
        )

        chunks = rebuild_index()

        print(
            f"RAG indexing completed: "
            f"{chunks} chunks"
        )

        return jsonify(
            {
                "success": True,

                "filename":
                    safe_filename,

                # Kept for frontend compatibility
                "document":
                    safe_filename,

                "chunks":
                    chunks,

                "message":
                    "PDF uploaded and RAG index created successfully."
            }
        )

    except Exception as error:

        print(
            "UPLOAD ERROR:",
            error
        )

        return jsonify(
            {
                "success": False,
                "error":
                    str(error)
            }
        ), 500


# =========================================================
# CHAT
# =========================================================

@app.route(
    "/api/chat",
    methods=["POST"]
)
def chat():

    try:

        data = request.get_json(
            silent=True
        ) or {}

        question = str(
            data.get(
                "question",
                ""
            )
        ).strip()

        if not question:

            return jsonify(
                {
                    "success": False,
                    "error":
                        "Please enter a question."
                }
            ), 400

        print(
            f"User question: "
            f"{question}"
        )

        answer = answer_question(
            question
        )

        return jsonify(
            {
                "success": True,

                "answer":
                    answer
            }
        )

    except Exception as error:

        print(
            "CHAT ERROR:",
            error
        )

        return jsonify(
            {
                "success": False,
                "error":
                    str(error)
            }
        ), 500


# =========================================================
# STATUS
# =========================================================

@app.route(
    "/api/status",
    methods=["GET"]
)
def status():

    try:

        pdf_files = list(
            DOCUMENTS_DIR.glob("*.pdf")
        )

        filename = (
            pdf_files[0].name
            if pdf_files
            else None
        )

        chunks = collection.count()

        return jsonify(
            {
                "success": True,

                "pdf_count":
                    len(pdf_files),

                "filename":
                    filename,

                # Frontend compatibility
                "document":
                    filename,

                "chunks":
                    chunks
            }
        )

    except Exception as error:

        print(
            "STATUS ERROR:",
            error
        )

        return jsonify(
            {
                "success": False,
                "error":
                    str(error)
            }
        ), 500


# =========================================================
# DELETE DOCUMENT
# =========================================================

@app.route(
    "/api/delete",
    methods=["DELETE"]
)
def delete_document():

    try:

        print(
            "Deleting current document..."
        )

        delete_all_pdfs()

        clear_collection()

        return jsonify(
            {
                "success": True,

                "message":
                    "Document deleted successfully.",

                "filename":
                    None,

                "document":
                    None,

                "chunks":
                    0
            }
        )

    except Exception as error:

        print(
            "DELETE ERROR:",
            error
        )

        return jsonify(
            {
                "success": False,
                "error":
                    str(error)
            }
        ), 500


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    print()
    print(
        "================================================="
    )
    print(
        "              RAG AI CHATBOT"
    )
    print(
        "================================================="
    )
    print(
        f"Generation model: "
        f"{GENERATION_MODEL}"
    )
    print(
        f"Embedding model: "
        f"{EMBEDDING_MODEL}"
    )
    print(
        f"Documents folder: "
        f"{DOCUMENTS_DIR}"
    )
    print(
        f"ChromaDB folder: "
        f"{CHROMA_DIR}"
    )
    print(
        "================================================="
    )
    print()

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False
    )