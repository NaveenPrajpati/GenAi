"""
RAG from Scratch - Manual RAG Pipeline (No Magic)
===================================================

This module implements a complete RAG pipeline manually, without using
high-level abstractions like RetrievalQA.

Requirements Met:
-----------------
✅ Text splitter (RecursiveCharacterTextSplitter)
✅ Embeddings (OpenAIEmbeddings)
✅ Vector store (Chroma - persistent) or InMemoryVectorStore
✅ Retriever (custom top-k retrieval)
✅ NO RetrievalQA - manual pipeline

Pipeline Steps:
---------------
1. Ingest text files → Document objects
2. Chunk documents → Smaller pieces with overlap
3. Embed chunks → Vector representations
4. Store in vector database
5. Retrieve top-k relevant chunks
6. Inject context into prompt
7. Generate answer with LLM
8. Return answer + source chunks

Architecture:
-------------
    Text Files
        │
        ▼
    ┌──────────────┐
    │ TextLoader   │  Step 1: Ingest
    └──────────────┘
        │
        ▼
    ┌──────────────┐
    │ TextSplitter │  Step 2: Chunk
    │ (recursive)  │
    └──────────────┘
        │
        ▼
    ┌──────────────┐
    │ Embeddings   │  Step 3: Embed
    │ (OpenAI)     │
    └──────────────┘
        │
        ▼
    ┌──────────────┐
    │ VectorStore  │  Step 4: Store
    │ (Chroma)     │
    └──────────────┘
        │
        │ Query
        ▼
    ┌──────────────┐
    │ Retriever    │  Step 5: Retrieve top-k
    │ (similarity) │
    └──────────────┘
        │
        ▼
    ┌──────────────┐
    │ Prompt +     │  Step 6-7: Context + Generate
    │ LLM Chain    │
    └──────────────┘
        │
        ▼
    ┌──────────────┐
    │ Answer +     │  Step 8: Output
    │ Sources      │
    └──────────────┘

Expected Output:
----------------
{
  "answer": "The answer to your question...",
  "sources": [
    {"content": "chunk text...", "metadata": {...}},
    {"content": "chunk text...", "metadata": {...}}
  ]
}

Updated for LangChain 1.0+ (2025-2026)
"""

from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel, Field
import os

load_dotenv()


# =============================================================================
# Step 1: Document Ingestion - Load Text Files
# =============================================================================

def load_text_files(file_paths: List[str] = None, directory: str = None) -> List[Document]:
    """
    Load text files into Document objects.

    Args:
        file_paths: List of specific file paths to load
        directory: Directory containing text files to load

    Returns:
        List of Document objects with content and metadata
    """
    documents = []

    if directory and os.path.exists(directory):
        # Load all .txt files from directory
        loader = DirectoryLoader(
            directory,
            glob="**/*.txt",
            loader_cls=TextLoader,
            loader_kwargs={"encoding": "utf-8"}
        )
        documents.extend(loader.load())

    if file_paths:
        for path in file_paths:
            if os.path.exists(path):
                loader = TextLoader(path, encoding="utf-8")
                documents.extend(loader.load())

    print(f"📄 Loaded {len(documents)} document(s)")
    return documents


# =============================================================================
# Step 2: Text Chunking - Split Documents into Smaller Pieces
# =============================================================================

def chunk_documents(
    documents: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 100
) -> List[Document]:
    """
    Split documents into smaller chunks with overlap.

    Args:
        documents: List of Document objects
        chunk_size: Maximum characters per chunk
        chunk_overlap: Overlap between consecutive chunks

    Returns:
        List of chunked Document objects
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]  # Priority order for splitting
    )

    chunks = text_splitter.split_documents(documents)

    # Add chunk index to metadata
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    print(f"📦 Created {len(chunks)} chunks (size={chunk_size}, overlap={chunk_overlap})")
    return chunks


# =============================================================================
# Step 3 & 4: Embedding + Vector Store
# =============================================================================

def create_vector_store(
    chunks: List[Document],
    persist_directory: str = None
) -> Chroma:
    """
    Create embeddings and store in vector database.

    Args:
        chunks: List of chunked Document objects
        persist_directory: Path to persist vector store (None for in-memory)

    Returns:
        Chroma vector store instance
    """
    # Initialize embedding model
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Create vector store
    if persist_directory:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print(f"💾 Vector store persisted to: {persist_directory}")
    else:
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings
        )
        print("🧠 Vector store created (in-memory)")

    return vectorstore


def load_existing_vector_store(persist_directory: str) -> Chroma:
    """Load an existing persisted vector store."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )


# =============================================================================
# Step 5: Retrieval - Get Top-K Relevant Chunks
# =============================================================================

def retrieve_chunks(
    vectorstore: Chroma,
    query: str,
    k: int = 4
) -> List[Document]:
    """
    Retrieve top-k most relevant chunks for a query.

    Args:
        vectorstore: Chroma vector store
        query: User question
        k: Number of chunks to retrieve

    Returns:
        List of relevant Document chunks
    """
    # Method 1: Using retriever interface
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )
    chunks = retriever.invoke(query)

    # Method 2: Direct similarity search (with scores)
    # chunks_with_scores = vectorstore.similarity_search_with_score(query, k=k)

    print(f"🔍 Retrieved {len(chunks)} relevant chunks")
    return chunks


def retrieve_with_scores(
    vectorstore: Chroma,
    query: str,
    k: int = 4
) -> List[tuple]:
    """
    Retrieve chunks with relevance scores.

    Returns:
        List of (Document, score) tuples
    """
    results = vectorstore.similarity_search_with_score(query, k=k)
    print(f"🔍 Retrieved {len(results)} chunks with scores")
    for doc, score in results:
        print(f"   Score: {score:.4f} - {doc.page_content[:50]}...")
    return results


# =============================================================================
# Step 6 & 7: Context Injection + Answer Generation
# =============================================================================

def format_context(chunks: List[Document]) -> str:
    """Format retrieved chunks into context string."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        context_parts.append(f"[Source {i}: {source}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(context_parts)


def create_rag_chain(vectorstore: Chroma, k: int = 4):
    """
    Create the complete RAG chain using LCEL.

    Returns:
        A runnable chain that takes a question and returns answer + sources
    """
    # Initialize LLM
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    parser = StrOutputParser()

    # Create retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # RAG Prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Only use information from the provided context
2. If the context doesn't contain the answer, say "I don't have enough information to answer this question"
3. Cite which source(s) you used in your answer
4. Be concise but thorough"""),
        ("human", """Context:
{context}

Question: {question}

Answer:""")
    ])

    # Build the chain manually (no RetrievalQA!)
    def retrieve_and_format(inputs: dict) -> dict:
        """Retrieve chunks and format them."""
        question = inputs["question"]
        chunks = retriever.invoke(question)
        return {
            "question": question,
            "context": format_context(chunks),
            "source_chunks": chunks  # Keep original chunks for output
        }

    # Complete RAG chain
    rag_chain = (
        RunnableLambda(retrieve_and_format)
        | RunnableParallel({
            "answer": rag_prompt | model | parser,
            "source_chunks": lambda x: x["source_chunks"]
        })
    )

    return rag_chain


# =============================================================================
# Step 8: Complete RAG Pipeline with Sources
# =============================================================================

class RAGResponse(BaseModel):
    """Structured response with answer and sources."""
    answer: str = Field(description="The generated answer")
    sources: List[Dict[str, Any]] = Field(description="Source chunks used")


def ask_question(
    vectorstore: Chroma,
    question: str,
    k: int = 4
) -> Dict[str, Any]:
    """
    Complete RAG pipeline: question → answer + sources.

    Args:
        vectorstore: Chroma vector store
        question: User question
        k: Number of chunks to retrieve

    Returns:
        Dict with 'answer' and 'sources'
    """
    # Create chain
    chain = create_rag_chain(vectorstore, k=k)

    # Run chain
    result = chain.invoke({"question": question})

    # Format response
    response = {
        "answer": result["answer"],
        "sources": [
            {
                "content": chunk.page_content,
                "metadata": chunk.metadata
            }
            for chunk in result["source_chunks"]
        ]
    }

    return response


# =============================================================================
# Alternative: Manual Step-by-Step RAG (Most Explicit)
# =============================================================================

def manual_rag_pipeline(
    vectorstore: Chroma,
    question: str,
    k: int = 4
) -> Dict[str, Any]:
    """
    Completely manual RAG - step by step, no abstractions.
    """
    # Step 5: Retrieve
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})
    chunks = retriever.invoke(question)

    # Step 6: Format context
    context = format_context(chunks)

    # Step 7: Generate answer
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    prompt = f"""You are a helpful assistant. Answer based on the context only.

Context:
{context}

Question: {question}

Answer:"""

    response = model.invoke(prompt)
    answer = response.content

    # Step 8: Return with sources
    return {
        "answer": answer,
        "sources": [
            {
                "content": chunk.page_content[:200] + "...",
                "metadata": chunk.metadata
            }
            for chunk in chunks
        ]
    }


# =============================================================================
# Demo: Sample Text Data
# =============================================================================

SAMPLE_DOCUMENTS = [
    """MongoDB Tutorial: Introduction to Collections

MongoDB stores data in collections. A collection is a group of MongoDB documents.
It is the equivalent of a table in relational databases.

To create a collection, use the createCollection() method:
db.createCollection("myCollection")

Collections are created automatically when you insert your first document.
""",
    """MongoDB Tutorial: Inserting Data

To add data to a MongoDB collection, use the insertOne() or insertMany() methods.

Insert a single document:
db.users.insertOne({
    name: "John Doe",
    email: "john@example.com",
    age: 30
})

Insert multiple documents:
db.users.insertMany([
    { name: "Alice", age: 25 },
    { name: "Bob", age: 35 }
])

The _id field is automatically added if not specified.
""",
    """MongoDB Tutorial: Querying Data

To retrieve data from MongoDB, use the find() method.

Find all documents:
db.users.find()

Find with conditions:
db.users.find({ age: { $gt: 25 } })

Find one document:
db.users.findOne({ name: "John Doe" })

Common query operators: $eq, $ne, $gt, $lt, $gte, $lte, $in, $nin
""",
    """MongoDB Tutorial: Updating Data

Use updateOne() or updateMany() to modify existing documents.

Update a single document:
db.users.updateOne(
    { name: "John Doe" },
    { $set: { age: 31 } }
)

Update multiple documents:
db.users.updateMany(
    { age: { $lt: 30 } },
    { $inc: { age: 1 } }
)

Update operators: $set, $unset, $inc, $push, $pull
"""
]


def create_sample_vectorstore() -> Chroma:
    """Create a vector store from sample documents."""
    # Create Document objects
    documents = [
        Document(
            page_content=content,
            metadata={"source": f"mongodb_tutorial_part_{i+1}.txt", "part": i+1}
        )
        for i, content in enumerate(SAMPLE_DOCUMENTS)
    ]

    # Chunk documents
    chunks = chunk_documents(documents, chunk_size=300, chunk_overlap=50)

    # Create vector store
    vectorstore = create_vector_store(chunks)

    return vectorstore


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RAG FROM SCRATCH - Manual Pipeline Demo")
    print("=" * 60)

    # ---------------------------------------------------------------------
    # Setup: Create vector store from sample data
    # ---------------------------------------------------------------------
    print("\n📌 Step 1-4: Ingesting and indexing documents...")
    vectorstore = create_sample_vectorstore()

    # ---------------------------------------------------------------------
    # Example 1: Basic RAG Query
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📌 Example 1: RAG Query with Sources")
    print("=" * 60)

    question = "How do I add data to a MongoDB collection?"
    print(f"\n❓ Question: {question}")

    result = ask_question(vectorstore, question, k=3)

    print(f"\n✅ Answer:\n{result['answer']}")

    print(f"\n📚 Sources ({len(result['sources'])} chunks):")
    for i, source in enumerate(result['sources'], 1):
        print(f"\n   Source {i}:")
        print(f"   File: {source['metadata'].get('source', 'unknown')}")
        print(f"   Content: {source['content'][:150]}...")

    # ---------------------------------------------------------------------
    # Example 2: Manual Pipeline (Most Explicit)
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📌 Example 2: Manual Step-by-Step RAG")
    print("=" * 60)

    question2 = "What query operators are available in MongoDB?"
    print(f"\n❓ Question: {question2}")

    manual_result = manual_rag_pipeline(vectorstore, question2, k=2)

    print(f"\n✅ Answer:\n{manual_result['answer']}")
    print(f"\n📚 Sources: {len(manual_result['sources'])} chunks returned")

    # ---------------------------------------------------------------------
    # Example 3: Retrieve with Scores
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📌 Example 3: Retrieval with Relevance Scores")
    print("=" * 60)

    question3 = "How to update documents?"
    print(f"\n❓ Question: {question3}")

    chunks_with_scores = retrieve_with_scores(vectorstore, question3, k=3)

    # ---------------------------------------------------------------------
    # Expected Output Format
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📌 Expected Output Format")
    print("=" * 60)
    print("""
{
  "answer": "To add data to a MongoDB collection, use insertOne() for a
            single document or insertMany() for multiple documents...",
  "sources": [
    {
      "content": "MongoDB Tutorial: Inserting Data...",
      "metadata": {"source": "mongodb_tutorial_part_2.txt", "part": 2}
    },
    ...
  ]
}
""")


# =============================================================================
# Summary: RAG from Scratch Key Points
# =============================================================================
"""
RAG Pipeline Components (No Magic):
-----------------------------------

1. DOCUMENT LOADING
   - TextLoader for single files
   - DirectoryLoader for batch loading
   - Always specify encoding

2. TEXT SPLITTING
   - RecursiveCharacterTextSplitter (best for general text)
   - Choose chunk_size based on context window
   - Use overlap to preserve context across chunks

3. EMBEDDINGS
   - OpenAIEmbeddings (text-embedding-3-small recommended)
   - Embed both documents and queries

4. VECTOR STORE
   - Chroma (persistent or in-memory)
   - Stores embeddings + original text + metadata

5. RETRIEVER
   - similarity_search for basic retrieval
   - similarity_search_with_score for debugging
   - as_retriever() for LCEL integration

6. CONTEXT INJECTION
   - Format chunks into readable context
   - Include source attribution

7. GENERATION
   - Use ChatPromptTemplate with system instructions
   - Tell model to only use provided context

8. SOURCES
   - Always return source chunks with answer
   - Include metadata (file, page, chunk_index)

Why No RetrievalQA?
-------------------
- More control over each step
- Better debugging capability
- Easier to customize (reranking, filtering, etc.)
- LCEL is more composable and modern

Pro Tips:
---------
- Chunk size: 500-1000 chars for most use cases
- Overlap: 10-20% of chunk size
- k: Start with 3-5, adjust based on context window
- Always return sources for transparency
"""
