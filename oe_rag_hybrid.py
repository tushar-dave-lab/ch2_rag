import os
import glob
import shutil
import argparse
import pdfplumber
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise EnvironmentError("GROQ_API_KEY not found. Check your .env file.")

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
DEFAULT_PROVIDER = "huggingface"  # "huggingface" or "openai"
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
DOCS_DIR = "./oe_docs"


def get_embedding_model(provider: str):
    if provider == "huggingface":
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        persist_dir = "./chroma_db_minilm"
    elif provider == "openai":
        embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small"
        )
        persist_dir = "./chroma_db_openai"
    else:
        raise ValueError("Provider must be 'huggingface' or 'openai'")

    return embeddings, persist_dir


# ---------------------------------------------------------
# 2. TABLE-AWARE PDF PARSER (pdfplumber -> Markdown)
# ---------------------------------------------------------
def table_to_markdown(table: list[list[str | None]]) -> str:
    """Converts a 2D list extracted by pdfplumber into a clean Markdown table."""
    if not table or not any(table):
        return ""

    cleaned = [
        [(cell or "").replace("\n", " ").strip() for cell in row]
        for row in table if any(row)
    ]
    if not cleaned:
        return ""

    header = "| " + " | ".join(cleaned[0]) + " |"
    separator = "| " + " | ".join(["---"] * len(cleaned[0])) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in cleaned[1:]]
    return "\n".join([header, separator] + rows)


def load_pdfs_with_tables(docs_dir: str) -> list[Document]:
    """Extracts narrative text and structured Markdown tables per PDF page."""
    pdf_files = glob.glob(os.path.join(docs_dir, "*.pdf"))
    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in {docs_dir}. Add your OE documents first."
        )

    documents = []
    for pdf_path in pdf_files:
        with pdfplumber.open(pdf_path) as pdf:
            for page_idx, page in enumerate(pdf.pages):
                tables = page.find_tables()
                markdown_tables = []
                table_bboxes = [t.bbox for t in tables]

                for t in tables:
                    md = table_to_markdown(t.extract())
                    if md:
                        markdown_tables.append(md)

                def not_inside_tables(obj):
                    if obj.get("object_type") != "char":
                        return True
                    x0, top, x1, bottom = obj["x0"], obj["top"], obj["x1"], obj["bottom"]
                    for bx0, btop, bx1, bbottom in table_bboxes:
                        if x0 >= bx0 and x1 <= bx1 and top >= btop and bottom <= bbottom:
                            return False
                    return True

                clean_page = page.filter(not_inside_tables)
                narrative_text = clean_page.extract_text() or ""

                combined_parts = []
                if narrative_text.strip():
                    combined_parts.append(narrative_text.strip())
                if markdown_tables:
                    combined_parts.append(
                        "\n\n### Extracted Plan Tables:\n" + "\n\n".join(markdown_tables)
                    )

                full_page_content = "\n\n".join(combined_parts)
                if full_page_content:
                    documents.append(
                        Document(
                            page_content=full_page_content,
                            metadata={"source": pdf_path, "page": page_idx},
                        )
                    )
    return documents


# ---------------------------------------------------------
# 3. INGESTION & HYBRID RETRIEVER (Chroma + BM25)
# ---------------------------------------------------------
def build_hybrid_retriever(embeddings, persist_dir: str, reindex: bool = False):
    """Builds or loads ChromaDB + BM25 and combines them via EnsembleRetriever."""
    if reindex and os.path.exists(persist_dir):
        print(f"--reindex flag detected. Removing old index at {persist_dir}...")
        shutil.rmtree(persist_dir)

    if os.path.exists(persist_dir) and os.listdir(persist_dir):
        print(f"Loading existing vector index from {persist_dir}...")
        vectorstore = Chroma(
            persist_directory=persist_dir, embedding_function=embeddings
        )

        stored_data = vectorstore.get(include=["documents", "metadatas"])
        chunks = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(stored_data["documents"], stored_data["metadatas"])
        ]
        print(f"Loaded {len(chunks)} chunks into BM25 keyword index.")
    else:
        print(f"Parsing Open Enrollment PDFs with table extraction from {DOCS_DIR}...")
        raw_docs = load_pdfs_with_tables(DOCS_DIR)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=250,
            separators=["\n### ", "\n\n", "\n", " ", ""],
        )
        chunks = text_splitter.split_documents(raw_docs)
        print(f"Parsed {len(raw_docs)} pages into {len(chunks)} table-aware chunks.")

        print(f"Generating embeddings and storing in {persist_dir}...")
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_dir,
        )

    vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = 5

    hybrid_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.4, 0.6],
    )
    return hybrid_retriever


# ---------------------------------------------------------
# 4. CONVERSATIONAL RAG CHAIN
# ---------------------------------------------------------
def create_conversational_rag(retriever, model_name: str):
    llm = ChatGroq(model=model_name, temperature=0)

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "Given a chat history and the latest user question which might reference context "
            "in the chat history, formulate a standalone question which can be understood "
            "without the chat history. Do NOT answer the question, just reformulate it if needed."
        )),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    qa_system_prompt = """You are an objective employee benefits assistant. 
Use ONLY the provided Open Enrollment document context (including Markdown tables) to answer the user's question.
When comparing plans (such as HDHP, PPO, or HMO), explicitly cite premiums, deductibles, 
out-of-pocket maximums, HSA/FSA employer contributions, and copays if present in the context.
Present side-by-side plan comparisons in a clean Markdown table when helpful.
If the documents do not contain enough information to answer, state what is missing.

Context:
{context}"""

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)


# ---------------------------------------------------------
# 5. STREAMING INTERACTIVE CHAT LOOP
# ---------------------------------------------------------
def run_chat_loop(rag_chain, model_name: str):
    chat_history = []
    print("\n" + "=" * 60)
    print(f"Open Enrollment Assistant Ready! (Model: {model_name})")
    print("Commands: 'clear' to reset chat history | 'exit' or 'quit' to stop")
    print("=" * 60)

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting...")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break
        if user_input.lower() == "clear":
            chat_history.clear()
            print("Chat history cleared.")
            continue

        print("\nAssistant: ", end="", flush=True)

        full_answer = []
        retrieved_docs = []

        for chunk in rag_chain.stream({"input": user_input, "chat_history": chat_history}):
            if "context" in chunk:
                retrieved_docs = chunk["context"]
            if "answer" in chunk:
                token = chunk["answer"]
                print(token, end="", flush=True)
                full_answer.append(token)

        print()

        if retrieved_docs:
            sources = sorted({
                f"{os.path.basename(doc.metadata.get('source', 'Unknown'))} (p. {doc.metadata.get('page', 0) + 1})"
                for doc in retrieved_docs
            })
            print(f"\n[Sources checked: {', '.join(sources)}]")

        complete_response = "".join(full_answer)
        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=complete_response),
        ])
        if len(chat_history) > 10:
            chat_history = chat_history[-10:]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open Enrollment Hybrid RAG Assistant")
    parser.add_argument(
        "--reindex",
        action="store_true",
        help="Delete existing vector database and re-ingest all PDFs in ./oe_docs",
    )
    parser.add_argument(
        "--provider",
        choices=["huggingface", "openai"],
        default=DEFAULT_PROVIDER,
        help="Choose embedding model provider (default: huggingface)",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_GROQ_MODEL,
        help=f"Groq model ID (default: {DEFAULT_GROQ_MODEL})",
    )
    args = parser.parse_args()

    os.makedirs(DOCS_DIR, exist_ok=True)
    embeddings, persist_dir = get_embedding_model(args.provider)
    hybrid_retriever = build_hybrid_retriever(
        embeddings, persist_dir, reindex=args.reindex
    )
    rag_chain = create_conversational_rag(hybrid_retriever, args.model)
    run_chat_loop(rag_chain, args.model)
