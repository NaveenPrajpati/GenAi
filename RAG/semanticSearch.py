"""
Semantic Search with Vector Embeddings
========================================

LEARNING OBJECTIVES:
- Understand vector embeddings and semantic search
- Load and process PDF documents
- Split documents into chunks
- Create and query vector stores
- Build a retrieval pipeline

CONCEPT:
Semantic search finds documents based on meaning, not just keywords.
It works by:
1. Converting text into numerical vectors (embeddings)
2. Storing vectors in a vector database
3. Converting queries to vectors
4. Finding similar vectors using distance metrics

VECTOR SEARCH FLOW:
    ┌─────────────────────────────────────────────────────────────┐
    │                   Semantic Search Flow                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │   Documents                     Query                        │
    │       │                           │                          │
    │       ▼                           ▼                          │
    │   ┌────────┐                 ┌────────┐                     │
    │   │ Split  │                 │Embedding│                     │
    │   │ Chunks │                 │ Model   │                     │
    │   └───┬────┘                 └────┬───┘                     │
    │       │                           │                          │
    │       ▼                           ▼                          │
    │   ┌────────┐                 [0.1, 0.3, ...]  (query vector) │
    │   │Embedding│                     │                          │
    │   │ Model   │                     │                          │
    │   └───┬────┘                      │                          │
    │       │                           │                          │
    │       ▼                           │                          │
    │   ┌────────────────┐              │                          │
    │   │  Vector Store  │◄─────────────┘                          │
    │   │   (Chroma)     │   similarity search                     │
    │   └───────┬────────┘                                         │
    │           │                                                  │
    │           ▼                                                  │
    │   Top K Similar Documents                                    │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

KEY COMPONENTS:
1. Document Loader: Reads files (PDF, text, etc.)
2. Text Splitter: Breaks documents into chunks
3. Embedding Model: Converts text to vectors
4. Vector Store: Stores and indexes vectors
5. Retriever: Searches for similar documents

PREREQUISITES:
- pip install langchain-chroma langchain-openai pypdf
- OpenAI API key in .env

NEXT STEPS:
- RAG/youtubeChatbot.py - Complete RAG chatbot
- practicsProjects/2ragPdf.py - Production RAG pipeline
"""

from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import chain
from typing import List
from dotenv import load_dotenv
import os

load_dotenv()

# =============================================================================
# STEP 1: Sample Documents
# =============================================================================
# Simple documents for demonstration

documents = [
    Document(
        page_content="Dogs are great companions, known for their loyalty and friendliness.",
        metadata={"source": "mammal-pets-doc"},
    ),
    Document(
        page_content="Cats are independent pets that often enjoy their own space.",
        metadata={"source": "mammal-pets-doc"},
    ),
]

# =============================================================================
# STEP 2: Load PDF (if available)
# =============================================================================
# Try to load a PDF file for more realistic demo

file_path = "../mongtutorial.pdf"  # Update path as needed
docs = []

if os.path.exists(file_path):
    print(f"Loading PDF from: {file_path}")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")
    if docs:
        print(f"First page preview: {docs[0].page_content[:200]}\n")
        print(f"Metadata: {docs[0].metadata}")
else:
    print(f"PDF not found at {file_path}, using sample documents only")
    docs = documents

# =============================================================================
# STEP 3: Split Documents into Chunks
# =============================================================================
# RecursiveCharacterTextSplitter intelligently splits text

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # Max characters per chunk
    chunk_overlap=200,    # Overlap between chunks for context continuity
    add_start_index=True  # Track position in original document
)
all_splits = text_splitter.split_documents(docs)
print(f"\nSplit into {len(all_splits)} chunks")

# =============================================================================
# STEP 4: Create Embeddings
# =============================================================================
# text-embedding-3-small is cost-effective, use text-embedding-3-large for better quality

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# Demonstrate embedding generation
if all_splits:
    vector_1 = embeddings.embed_query(all_splits[0].page_content)
    print(f"\nGenerated vector of length {len(vector_1)}")
    print(f"First 5 values: {vector_1[:5]}")

# =============================================================================
# STEP 5: Create Vector Store
# =============================================================================
# Chroma is a lightweight vector database, great for development

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",  # Persist to disk
)

# Add documents to vector store
if all_splits:
    ids = vector_store.add_documents(documents=all_splits)
    print(f"\nAdded {len(ids)} documents to vector store")

# =============================================================================
# STEP 6: Similarity Search
# =============================================================================
# Search for documents similar to a query

query = "What are the key features?"
results = vector_store.similarity_search(query, k=2)

print(f"\n--- Similarity Search Results ---")
print(f"Query: '{query}'")
for i, doc in enumerate(results):
    print(f"\nResult {i+1}:")
    print(f"  Content: {doc.page_content[:150]}...")
    print(f"  Metadata: {doc.metadata}")

# =============================================================================
# STEP 7: Similarity Search with Scores
# =============================================================================
# Get relevance scores along with results

results_with_scores = vector_store.similarity_search_with_score(query, k=2)

print(f"\n--- Results with Scores ---")
for doc, score in results_with_scores:
    print(f"\nScore: {score:.4f} (lower = more similar)")
    print(f"Content: {doc.page_content[:100]}...")

# =============================================================================
# STEP 8: Custom Retriever with @chain
# =============================================================================
# Create a custom retriever function


@chain
def custom_retriever(query: str) -> List[Document]:
    """Custom retriever that returns top 2 similar documents."""
    return vector_store.similarity_search(query, k=2)


# Test custom retriever
print("\n--- Custom Retriever ---")
custom_results = custom_retriever.invoke("Tell me about the document")
for doc in custom_results:
    print(f"  - {doc.page_content[:80]}...")

# Batch processing
print("\n--- Batch Retrieval ---")
batch_results = custom_retriever.batch([
    "What is the main topic?",
    "Key concepts mentioned?",
])
for i, results in enumerate(batch_results):
    print(f"\nQuery {i+1} results: {len(results)} documents found")

# =============================================================================
# STEP 9: Create Standard Retriever
# =============================================================================
# Convert vector store to a retriever interface

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 2},
)

print("\n--- Standard Retriever ---")
retrieved_docs = retriever.invoke("main topic")
print(f"Retrieved {len(retrieved_docs)} documents")

# =============================================================================
# BEST PRACTICES
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SEMANTIC SEARCH BEST PRACTICES")
    print("=" * 60)
    print("""
    1. CHUNK SIZE: 500-1000 characters works well for most cases

    2. CHUNK OVERLAP: Use 10-20% overlap for context continuity

    3. EMBEDDING MODELS:
       - text-embedding-3-small: Fast, cheap ($0.02/1M tokens)
       - text-embedding-3-large: Higher quality ($0.13/1M tokens)

    4. VECTOR STORES:
       - Chroma: Great for development, small datasets
       - FAISS: Fast, good for medium datasets
       - Pinecone/Weaviate: Production, large scale

    5. SEARCH PARAMETERS:
       - k: Start with 3-5, adjust based on needs
       - score_threshold: Filter low-relevance results

    6. METADATA: Add source, date, author for filtering

    7. HYBRID SEARCH: Combine with BM25 for better results
    """)
