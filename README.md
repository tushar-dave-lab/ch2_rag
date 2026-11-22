# Open Enrollment Hybrid RAG Assistant (`oe_rag_hybrid.py`)

A conversational Retrieval-Augmented Generation (RAG) assistant in Python designed to parse Open Enrollment (OE) PDF documents—including multi-column medical plan comparison tables—and answer personalized benefits questions using **hybrid search (BM25 + ChromaDB)** and **Groq** token streaming.

## Features

* **Table-Aware PDF Extraction (`pdfplumber`):** Detects table bounding boxes on each PDF page, converts them into clean Markdown tables (`| Column | ... |`), and filters those coordinates out of the narrative body text so plan figures aren't duplicated or scrambled across columns.
* **Hybrid Retrieval (`EnsembleRetriever`):** Combines sparse keyword matching (`BM25Retriever`, weight `0.4`) for exact drug tiers, plan codes, and dollar figures with dense semantic search (`ChromaDB`, weight `0.6`) using Reciprocal Rank Fusion.
* **Flexible Embedding Providers:** Supports local, zero-cost embeddings via `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions) or cloud embeddings via OpenAI's `text-embedding-3-small` (1,536 dimensions) in separate persistent vector directories.
* **Conversational Memory & Streaming:** Rewrites follow-up questions using chat history (`create_history_aware_retriever`), streams responses token-by-token via Groq (`openai/gpt-oss-120b` by default), and prints source PDF filenames and page numbers after every answer.

---

## Project Structure

```text
ch2_rag/
├── .env                  # API keys (GROQ_API_KEY, optional OPENAI_API_KEY)
├── pyproject.toml        # uv project configuration
├── oe_rag_hybrid.py      # Main RAG ingestion & streaming chat script
├── README.md             # Project documentation
├── oe_docs/              # Place your Open Enrollment PDFs here
└── chroma_db_minilm/     # Auto-generated local vector store (or chroma_db_openai/)
```

---

## Setup & Installation

### 1. Initialize Environment with `uv`
```bash
uv init --no-readme
uv add langchain langchain-classic langchain-community langchain-text-splitters langchain-huggingface langchain-openai langchain-groq langchain-chroma sentence-transformers pdfplumber rank_bm25 python-dotenv
```

### 2. Configure Environment Variables
Create a `.env` file in the root directory:

```env
GROQ_API_KEY="gsk_your_groq_api_key_here"

# Optional: Override default Groq model (defaults to openai/gpt-oss-120b)
# GROQ_MODEL="openai/gpt-oss-20b"

# Optional: Only required if running with --provider openai
# OPENAI_API_KEY="sk_your_openai_api_key_here"
```

### 3. Add Open Enrollment Documents
Place your benefits guides, Summary of Benefits and Coverage (SBC) PDFs, and plan comparison sheets inside the `./oe_docs` directory.

---

## Usage

### Start the Interactive Chat (Default: Local Hugging Face Embeddings + Groq)
```bash
uv run oe_rag_hybrid.py
```

### Rebuild the Index After Updating PDFs (`--reindex`)
```bash
uv run oe_rag_hybrid.py --reindex
```

### Switch Embedding Provider (`--provider`)
Use OpenAI's `text-embedding-3-small` instead of local `all-MiniLM-L6-v2`:
```bash
uv run oe_rag_hybrid.py --provider openai --reindex
```

### Override the Groq LLM (`--model`)
```bash
uv run oe_rag_hybrid.py --model openai/gpt-oss-20b
```

### In-Chat Commands
* `clear` — Reset the conversation history (retains the last 5 turns by default).
* `exit` or `quit` — Exit the assistant.

---

## Example Prompts

* *"I rarely visit the doctor, take one generic prescription monthly, and want to maximize tax-advantaged savings. Compare the HDHP and PPO plans based on my needs, including deductibles, HSA match, and annual out-of-pocket exposure."*
* *"Put the in-network deductibles, out-of-pocket maximums, and primary care copays for all medical plans into a side-by-side table."*
* *"What changes if I switch from Employee Only to Employee + Spouse coverage on the HDHP?"*
