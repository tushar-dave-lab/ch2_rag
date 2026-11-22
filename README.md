# Open Enrollment Hybrid RAG Assistant (`ch2_rag`)

A full-stack RAG assistant built with **Python, LangChain, FastAPI, and React** for parsing Open Enrollment PDFs, featuring hybrid search (BM25 + ChromaDB), Groq token streaming, and an interactive React frontend.

## Key Features & Architecture
* **Table-Aware Extraction (`pdfplumber`):** Extracts tabular data into clean Markdown tables while filtering out bounding boxes from narrative text.
* **Hybrid Retrieval (`EnsembleRetriever`):** Combines sparse BM25 keyword search (weight `0.4`) with dense ChromaDB semantic search (weight `0.6`).
* **Flexible Embeddings & Backend:** Supports local (`sentence-transformers/all-MiniLM-L6-v2`) or OpenAI embeddings, alongside FastAPI endpoints for SSE streaming (`/ask/stream`), live PDF uploads (`/upload`), and document management.
* **Project Structure:** Includes backend modules (`oe_rag_hybrid.py`, `api.py`), local vector stores (`chroma_db_minilm/`), and a Vite + React frontend (`frontend/`).

## Setup & Running the App
1. **Backend & Dependencies:** Initialize with `uv init --no-readme` and install required LangChain, FastAPI, and embedding packages.
2. **Environment:** Copy `.env.example` to `.env` and configure your `GROQ_API_KEY`.
3. **Frontend:** Run `npm install` inside the `frontend/` directory.
4. **Execution:** 
   * *Full-Stack:* Run `uv run uvicorn api:app --reload --port 8000` in one tab and `npm run dev` inside `frontend/` (at `http://localhost:5173`).
   * *CLI Mode:* Run `uv run oe_rag_hybrid.py` for terminal-based chat.

## API Endpoints
* `GET /health` – Check API readiness and active models.
* `GET /documents` – List indexed PDF files in `./oe_docs`.
* `POST /upload` – Upload and auto-index a new PDF.
* `POST /ask` & `POST /ask/stream` – Query endpoints (JSON or SSE stream).
* `POST /reindex` – Rebuild the ChromaDB and BM25 index.
