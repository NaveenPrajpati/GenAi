from __future__ import annotations

import os
from typing import List, Dict, Any, Tuple

# Loaders / docs
from langchain_community.document_loaders import (
    PyPDFLoader,
)  # or PyPDFium2Loader / PyMuPDFLoader
from langchain_core.documents import Document

# Splitters
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Embeddings (choose ONE of these)
# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

# Vector stores
from langchain_chroma import Chroma
from langchain_community.vectorstores.faiss import FAISS

# LLM + prompts + runnables
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
EMBEDDING_BACKEND = "hf"  # "hf" or "openai"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "pdf_rag"


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def load_pdfs(pdf_paths: List[str]) -> List[Document]:
    """Load and annotate PDFs (1-index page numbers)."""
    documents: List[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()  # returns List[Document]
        print("document -", len(docs))
        for d in docs:
            d.metadata.setdefault("source_file", os.path.basename(path))
            # PyPDFLoader uses 0-indexed page in metadata; normalize to 1-index for display
            if "page" in d.metadata:
                d.metadata["page"] = int(d.metadata["page"]) + 1
        documents.extend(docs)
    return documents


def chunk_documents(docs: List[Document]) -> List[Document]:
    """Recommended robust splitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        add_start_index=True,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(docs)


def get_embeddings():
    """Choose a single embedding model."""
    if EMBEDDING_BACKEND == "openai":
        # Requires OPENAI_API_KEY in env; model defaults to text-embedding-3-large/small
        from langchain_openai import OpenAIEmbeddings  # local import to avoid unused

        return OpenAIEmbeddings(model="text-embedding-3-large")  # or -small
    else:
        # Fast, strong default
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )


def build_vectorstore(chunks: List[Document], backend: str = "chroma"):
    """
    Create and populate a vector store using the modern constructors.
    backend: "chroma" (persistent) or "faiss" (in-memory / savable to disk).
    """
    embeddings = get_embeddings()

    if backend == "chroma":
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            persist_directory=PERSIST_DIR,
        )
        # Chroma persists automatically on add; no extra call needed
        return vs

    if backend == "faiss":
        vs = FAISS.from_documents(documents=chunks, embedding=embeddings)
        # Optionally persist FAISS to disk
        vs.save_local("./faiss_index")
        return vs

    raise ValueError("backend must be 'chroma' or 'faiss'")


def _format_docs(docs: List[Document]) -> str:
    """Plain-text context joiner for the prompt."""
    return "\n\n".join(d.page_content for d in docs)


def create_rag_chain(vectorstore) -> Any:
    """
    Runnable RAG pipeline that returns BOTH the answer and the retrieved docs.
    """
    retriever = vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 4}
    )

    # Use the community RAG prompt (or your own)
    prompt = hub.pull("rlm/rag-prompt")

    llm = ChatOpenAI(temperature=0)

    # Simplified approach - build the chain step by step
    def rag_chain(question: str):
        # Get relevant documents
        docs = retriever.invoke(question)

        # Format context
        context = _format_docs(docs)

        # Create prompt input
        prompt_input = {"context": context, "question": question}

        # Get response
        response = (prompt | llm | StrOutputParser()).invoke(prompt_input)

        return {"answer": response, "docs": docs}

    return rag_chain


def enhance_with_citations(answer: str, docs: List[Document]) -> Dict[str, Any]:
    """Append [Page N] style citations and return structured payload."""
    pages = []
    sources = []
    for d in docs:
        page = d.metadata.get("page", "Unknown")
        src = d.metadata.get("source_file", "Unknown")
        pages.append(page)
        sources.append(f"{src} (Page {page})")

    # Add a citation footer if none present
    if pages and ("[Page" not in answer and "[Pages" not in answer):
        unique = sorted({str(p) for p in pages})
        answer = f"{answer} [Pages: {', '.join(unique)}]"

    return {"answer": answer, "sources": sorted(set(sources)), "source_documents": docs}


# -----------------------------------------------------------------------------
# (Toy) Evaluation utilities
# -----------------------------------------------------------------------------
def evaluate_answer_quality(answer: str, expected: str) -> bool:
    a, e = answer.lower(), expected.lower()
    common = set(a.split()) & set(e.split())
    return len(common) / max(1, len(set(e.split()))) > 0.3


def evaluate_rag_system(chain, test_questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct = 0
    cited = 0
    detailed = []

    for item in test_questions:
        q = item["question"]
        expected = item["expected_answer"]

        out = chain.invoke(q)  # {"answer": str, "docs": List[Document]}
        enriched = enhance_with_citations(out["answer"], out["docs"])

        is_correct = evaluate_answer_quality(enriched["answer"], expected)
        has_cite = "[Page" in enriched["answer"] or "[Pages" in enriched["answer"]
        correct += int(is_correct)
        cited += int(has_cite)

        detailed.append(
            {
                "question": q,
                "answer": enriched["answer"],
                "correct": is_correct,
                "has_citation": has_cite,
                "sources": enriched["sources"],
            }
        )

    total = max(1, len(test_questions))
    return {
        "accuracy": correct / total,
        "citation_coverage": cited / total,
        "detailed_results": detailed,
    }


# -----------------------------------------------------------------------------
# Example main
# -----------------------------------------------------------------------------
def main():
    pdf_paths = ["/Users/naveen/Desktop/web/genAi/mongtutorial.pdf"]

    docs = load_pdfs(pdf_paths)
    chunks = chunk_documents(docs)

    # Pick one: "chroma" (persistent) or "faiss" (local)
    vectorstore = build_vectorstore(chunks, backend="chroma")

    qa_chain = create_rag_chain(vectorstore)

    # Simple REPL
    while True:
        q = input("Ask a question (or 'quit'): ").strip()
        if q.lower() == "quit":
            break

        try:
            # Call the chain with the question
            out = qa_chain(q)  # Now it's a function, not a runnable
            enriched = enhance_with_citations(out["answer"], out["docs"])
            print("\nAnswer:", enriched["answer"])
            print("Sources:", ", ".join(enriched["sources"]))
            print("-" * 60)
        except Exception as e:
            print(f"Error processing question: {e}")
            print("-" * 60)


if __name__ == "__main__":
    main()
