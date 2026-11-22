import os
import json
import glob
import shutil
from contextlib import asynccontextmanager
from typing import Literal
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage, AIMessage

from oe_rag_hybrid import (
    DEFAULT_PROVIDER,
    DEFAULT_GROQ_MODEL,
    DOCS_DIR,
    get_embedding_model,
    build_hybrid_retriever,
    create_conversational_rag,
)

app_state = {}


def _reload_rag_chain(reindex: bool = False):
    embeddings = app_state["embeddings"]
    persist_dir = app_state["persist_dir"]
    model_name = app_state["model_name"]
    retriever = build_hybrid_retriever(embeddings, persist_dir, reindex=reindex)
    app_state["rag_chain"] = create_conversational_rag(retriever, model_name)


@asynccontextmanager
async def lifespan(app: FastAPI):
    os.makedirs(DOCS_DIR, exist_ok=True)
    provider = os.getenv("EMBEDDING_PROVIDER", DEFAULT_PROVIDER)
    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)
    embeddings, persist_dir = get_embedding_model(provider)

    app_state["embeddings"] = embeddings
    app_state["persist_dir"] = persist_dir
    app_state["model_name"] = model_name

    if glob.glob(os.path.join(DOCS_DIR, "*.pdf")) or (
        os.path.exists(persist_dir) and os.listdir(persist_dir)
    ):
        _reload_rag_chain(reindex=False)

    yield
    app_state.clear()


app = FastAPI(
    title="Open Enrollment Hybrid RAG API",
    version="1.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    chat_history: list[ChatMessage] = Field(default_factory=list)


class AskResponse(BaseModel):
    answer: str
    sources: list[str]


def _to_langchain_messages(history: list[ChatMessage]):
    messages = []
    for msg in history[-10:]:
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))
    return messages


def _format_sources(docs) -> list[str]:
    return sorted({
        f"{os.path.basename(doc.metadata.get('source', 'Unknown'))} (p. {doc.metadata.get('page', 0) + 1})"
        for doc in docs
    })


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "ready": "rag_chain" in app_state,
        "model": app_state.get("model_name"),
    }


@app.get("/documents")
def list_documents():
    """Lists all currently indexed PDF filenames in ./oe_docs."""
    pdf_files = sorted(
        os.path.basename(p) for p in glob.glob(os.path.join(DOCS_DIR, "*.pdf"))
    )
    return {"documents": pdf_files}


@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Uploads a new Open Enrollment PDF into ./oe_docs and re-indexes the RAG pipeline."""
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only .pdf files are supported.")

    safe_name = os.path.basename(file.filename)
    dest_path = os.path.join(DOCS_DIR, safe_name)

    with open(dest_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        _reload_rag_chain(reindex=True)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {exc}")

    return {
        "status": "uploaded_and_indexed",
        "filename": safe_name,
        "documents": list_documents()["documents"],
    }


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest):
    rag_chain = app_state.get("rag_chain")
    if not rag_chain:
        raise HTTPException(
            status_code=503,
            detail="No Open Enrollment PDFs indexed yet. Upload a PDF first.",
        )

    result = rag_chain.invoke({
        "input": payload.question,
        "chat_history": _to_langchain_messages(payload.chat_history),
    })

    return AskResponse(
        answer=result["answer"],
        sources=_format_sources(result.get("context", [])),
    )


@app.post("/ask/stream")
def ask_question_stream(payload: AskRequest):
    """Server-Sent Events (SSE) endpoint streaming typed JSON events."""
    rag_chain = app_state.get("rag_chain")
    if not rag_chain:
        raise HTTPException(
            status_code=503,
            detail="No Open Enrollment PDFs indexed yet. Upload a PDF first.",
        )

    lc_history = _to_langchain_messages(payload.chat_history)

    def sse_generator():
        retrieved_docs = []
        try:
            for chunk in rag_chain.stream({"input": payload.question, "chat_history": lc_history}):
                if "context" in chunk:
                    retrieved_docs = chunk["context"]
                if "answer" in chunk:
                    event_payload = json.dumps({"type": "token", "content": chunk["answer"]})
                    yield f"data: {event_payload}\n\n"

            sources = _format_sources(retrieved_docs)
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/reindex")
def reindex_documents():
    if not glob.glob(os.path.join(DOCS_DIR, "*.pdf")):
        raise HTTPException(status_code=400, detail="No PDFs found in ./oe_docs to index.")
    _reload_rag_chain(reindex=True)
    return {"status": "reindexed", "persist_dir": app_state.get("persist_dir")}
