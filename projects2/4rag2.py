"""
RAG with Query Rewriting - Improved Retrieval Quality
======================================================

This module demonstrates how to improve RAG retrieval quality using
query rewriting. Ambiguous or contextual queries are transformed into
standalone, searchable questions before retrieval.

Requirements Met:
-----------------
✅ Two-step pipeline (rewrite → retrieve)
✅ Uses Runnables (LCEL composition)
✅ Shows original query, rewritten query, and retrieval results

Problem Solved:
---------------
Vague queries like "How does it work?" fail in RAG because:
1. They lack context (what is "it"?)
2. They don't contain searchable keywords
3. Vector similarity matching performs poorly

Solution: Query Rewriting
-------------------------
Transform ambiguous queries into standalone, specific questions
that contain all necessary context for effective retrieval.

Architecture:
-------------
    User Query: "How does it work?"
           │
           ▼
    ┌──────────────────────┐
    │  Query Rewriter      │  Step 1: Rewrite
    │  (LLM + Context)     │
    └──────────────────────┘
           │
           ▼
    Rewritten: "How does MongoDB's insertOne() method work
                for adding data to collections?"
           │
           ▼
    ┌──────────────────────┐
    │  Retriever           │  Step 2: Retrieve
    │  (Vector Search)     │
    └──────────────────────┘
           │
           ▼
    ┌──────────────────────┐
    │  RAG Chain           │  Step 3: Generate
    │  (Context + LLM)     │
    └──────────────────────┘
           │
           ▼
    Final Answer + Sources

Expected Output:
----------------
{
  "original_query": "How does it work?",
  "rewritten_query": "How does MongoDB's insertOne() method work...",
  "answer": "...",
  "sources": [...]
}

Updated for LangChain 1.0+ (2025-2026)
"""

from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough,
    RunnableLambda,
    RunnableSequence
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional
import os

load_dotenv()


# =============================================================================
# Initialize Components
# =============================================================================

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()


# =============================================================================
# Step 1: Query Rewriting
# =============================================================================

# Prompt for rewriting ambiguous queries into standalone questions
QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriting assistant. Your task is to transform
ambiguous or contextual queries into clear, standalone questions that can be
used for document retrieval.

Rules:
1. Make the query self-contained (no pronouns like "it", "this", "that" without context)
2. Add relevant keywords that would appear in documentation
3. Preserve the user's intent
4. If the query is already clear and specific, return it unchanged
5. Output ONLY the rewritten query, nothing else

Examples:
- "How does it work?" → "How does the system/feature work?" (needs context)
- "What are the benefits?" → "What are the benefits of [topic]?"
- "Can you explain more?" → "Can you provide more details about [topic]?"
"""),
    ("human", """Chat History (for context):
{chat_history}

Current Query: {query}

Rewritten Query:""")
])


def create_query_rewriter(chat_history: List[tuple] = None):
    """
    Create a query rewriting chain.

    Args:
        chat_history: List of (role, message) tuples for context

    Returns:
        A runnable that rewrites queries
    """
    # Format chat history for the prompt
    def format_history(history: List[tuple]) -> str:
        if not history:
            return "No previous conversation."

        formatted = []
        for role, message in history:
            formatted.append(f"{role.capitalize()}: {message}")
        return "\n".join(formatted[-5:])  # Last 5 messages for context

    history_str = format_history(chat_history or [])

    # Create the rewriting chain
    rewrite_chain = (
        RunnablePassthrough.assign(
            chat_history=lambda _: history_str
        )
        | QUERY_REWRITE_PROMPT
        | model
        | parser
    )

    return rewrite_chain


# =============================================================================
# Contextual Query Rewriter (with conversation context)
# =============================================================================

CONTEXTUAL_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a query rewriting assistant for a RAG system.

Your job is to rewrite the user's follow-up question into a standalone question
that can be understood without the conversation history.

Guidelines:
1. Replace pronouns (it, this, that, they) with the actual subjects from context
2. Include key terms from the conversation that are relevant to the query
3. Make the question specific and searchable
4. Keep the query concise but complete
5. Output ONLY the rewritten question

If the question is already standalone and clear, return it unchanged."""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{query}")
])


def create_contextual_rewriter():
    """
    Create a rewriter that uses MessagesPlaceholder for chat history.
    More elegant for multi-turn conversations.
    """
    return CONTEXTUAL_REWRITE_PROMPT | model | parser


# =============================================================================
# Step 2: RAG Pipeline with Query Rewriting
# =============================================================================

def format_context(chunks: List[Document]) -> str:
    """Format retrieved chunks into context string."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        source = chunk.metadata.get("source", "unknown")
        context_parts.append(f"[Source {i}: {source}]\n{chunk.page_content}")
    return "\n\n---\n\n".join(context_parts)


def create_rag_with_rewriting(vectorstore: Chroma, k: int = 4):
    """
    Create a complete RAG chain with query rewriting.

    Pipeline:
    1. Preserve original query
    2. Rewrite query for better retrieval
    3. Retrieve using rewritten query
    4. Generate answer
    5. Return all components
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # RAG Answer Prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Only use information from the provided context
2. If the context doesn't contain the answer, say so
3. Cite sources when possible
4. Be concise but thorough"""),
        ("human", """Context:
{context}

Question: {rewritten_query}

Answer:""")
    ])

    def process_with_rewriting(inputs: dict) -> dict:
        """
        Complete pipeline: rewrite → retrieve → format
        """
        original_query = inputs["query"]
        chat_history = inputs.get("chat_history", [])

        # Step 1: Rewrite the query
        rewriter = create_query_rewriter(chat_history)
        rewritten_query = rewriter.invoke({"query": original_query})

        # Step 2: Retrieve using rewritten query
        chunks = retriever.invoke(rewritten_query)

        # Step 3: Format context
        context = format_context(chunks)

        return {
            "original_query": original_query,
            "rewritten_query": rewritten_query,
            "context": context,
            "source_chunks": chunks
        }

    # Complete RAG chain with rewriting
    rag_chain = (
        RunnableLambda(process_with_rewriting)
        | RunnableParallel({
            "original_query": lambda x: x["original_query"],
            "rewritten_query": lambda x: x["rewritten_query"],
            "answer": rag_prompt | model | parser,
            "source_chunks": lambda x: x["source_chunks"]
        })
    )

    return rag_chain


# =============================================================================
# Alternative: Pure LCEL Query Rewriting Chain
# =============================================================================

def create_pure_lcel_rewrite_chain(vectorstore: Chroma, k: int = 4):
    """
    Alternative implementation using pure LCEL without lambdas in process function.
    Demonstrates more Runnable-native approach.
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Simple rewrite prompt (no chat history)
    simple_rewrite_prompt = ChatPromptTemplate.from_template(
        """Rewrite this query to be more specific and standalone for document search.
If already clear, return unchanged. Output ONLY the rewritten query.

Query: {query}
Rewritten:"""
    )

    # Rewrite chain
    rewrite_chain = simple_rewrite_prompt | model | parser

    # RAG prompt
    rag_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based on the context provided. Be concise."),
        ("human", "Context:\n{context}\n\nQuestion: {rewritten_query}\n\nAnswer:")
    ])

    # Build the chain using LCEL
    chain = (
        # Step 1: Pass through original and rewrite
        RunnablePassthrough.assign(
            rewritten_query=lambda x: rewrite_chain.invoke({"query": x["query"]})
        )
        # Step 2: Retrieve using rewritten query
        | RunnablePassthrough.assign(
            docs=lambda x: retriever.invoke(x["rewritten_query"])
        )
        # Step 3: Format context
        | RunnablePassthrough.assign(
            context=lambda x: format_context(x["docs"])
        )
        # Step 4: Generate answer
        | RunnableParallel({
            "original_query": lambda x: x["query"],
            "rewritten_query": lambda x: x["rewritten_query"],
            "answer": rag_prompt | model | parser,
            "sources": lambda x: [
                {"content": d.page_content[:200], "metadata": d.metadata}
                for d in x["docs"]
            ]
        })
    )

    return chain


# =============================================================================
# Comparison: With vs Without Query Rewriting
# =============================================================================

def compare_retrieval_quality(
    vectorstore: Chroma,
    query: str,
    chat_history: List[tuple] = None,
    k: int = 3
) -> Dict[str, Any]:
    """
    Compare retrieval results with and without query rewriting.

    Returns:
        Dict showing original vs rewritten query and their retrieval results
    """
    retriever = vectorstore.as_retriever(search_kwargs={"k": k})

    # Without rewriting
    original_results = vectorstore.similarity_search_with_score(query, k=k)

    # With rewriting
    rewriter = create_query_rewriter(chat_history)
    rewritten_query = rewriter.invoke({"query": query})
    rewritten_results = vectorstore.similarity_search_with_score(rewritten_query, k=k)

    return {
        "original_query": query,
        "rewritten_query": rewritten_query,
        "original_retrieval": [
            {
                "content": doc.page_content[:100] + "...",
                "score": float(score),
                "source": doc.metadata.get("source", "unknown")
            }
            for doc, score in original_results
        ],
        "rewritten_retrieval": [
            {
                "content": doc.page_content[:100] + "...",
                "score": float(score),
                "source": doc.metadata.get("source", "unknown")
            }
            for doc, score in rewritten_results
        ]
    }


# =============================================================================
# Demo: Sample Data
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
""",
    """MongoDB Tutorial: Indexing for Performance

Indexes improve query performance by allowing MongoDB to quickly locate documents.

Create an index:
db.users.createIndex({ email: 1 })  // Ascending index

Create a compound index:
db.users.createIndex({ name: 1, age: -1 })  // Multiple fields

View indexes:
db.users.getIndexes()

Drop an index:
db.users.dropIndex("email_1")

Indexes speed up reads but slow down writes. Use them strategically.
"""
]


def create_sample_vectorstore() -> Chroma:
    """Create a vector store from sample documents."""
    documents = [
        Document(
            page_content=content,
            metadata={"source": f"mongodb_tutorial_part_{i+1}.txt", "part": i+1}
        )
        for i, content in enumerate(SAMPLE_DOCUMENTS)
    ]

    # Chunk documents
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )
    chunks = text_splitter.split_documents(documents)

    # Add chunk index
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    # Create vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    print(f"📦 Created vector store with {len(chunks)} chunks")
    return vectorstore


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RAG WITH QUERY REWRITING - Improved Retrieval Demo")
    print("=" * 70)

    # Setup
    print("\n📌 Setting up vector store...")
    vectorstore = create_sample_vectorstore()

    # -------------------------------------------------------------------------
    # Example 1: Ambiguous Query Without Context
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📌 Example 1: Ambiguous Query (No Context)")
    print("=" * 70)

    ambiguous_query = "How does it work?"
    print(f"\n❓ Original Query: \"{ambiguous_query}\"")
    print("   (This query is vague - 'it' has no reference)")

    # Compare with and without rewriting
    comparison = compare_retrieval_quality(vectorstore, ambiguous_query, k=2)

    print(f"\n🔄 Rewritten Query: \"{comparison['rewritten_query']}\"")

    print("\n📊 Retrieval Comparison:")
    print("\n   WITHOUT Rewriting (Original Query):")
    for i, result in enumerate(comparison['original_retrieval'], 1):
        print(f"   {i}. Score: {result['score']:.4f} | {result['content'][:60]}...")

    print("\n   WITH Rewriting (Improved Query):")
    for i, result in enumerate(comparison['rewritten_retrieval'], 1):
        print(f"   {i}. Score: {result['score']:.4f} | {result['content'][:60]}...")

    # -------------------------------------------------------------------------
    # Example 2: Query with Conversation Context
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📌 Example 2: Follow-up Query with Conversation Context")
    print("=" * 70)

    # Simulate a conversation
    chat_history = [
        ("user", "Tell me about MongoDB insertOne method"),
        ("assistant", "The insertOne() method adds a single document to a collection..."),
        ("user", "What about adding multiple documents?"),
        ("assistant", "You can use insertMany() to add multiple documents at once...")
    ]

    followup_query = "How does it work?"

    print("\n📝 Conversation History:")
    for role, msg in chat_history:
        print(f"   {role.capitalize()}: {msg[:50]}...")

    print(f"\n❓ Follow-up Query: \"{followup_query}\"")

    # Rewrite with context
    comparison_with_context = compare_retrieval_quality(
        vectorstore,
        followup_query,
        chat_history=chat_history,
        k=2
    )

    print(f"\n🔄 Rewritten Query: \"{comparison_with_context['rewritten_query']}\"")

    print("\n📊 Retrieval Results (With Context-Aware Rewriting):")
    for i, result in enumerate(comparison_with_context['rewritten_retrieval'], 1):
        print(f"   {i}. Score: {result['score']:.4f}")
        print(f"      Source: {result['source']}")
        print(f"      Content: {result['content']}")

    # -------------------------------------------------------------------------
    # Example 3: Full RAG Pipeline with Rewriting
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📌 Example 3: Complete RAG with Query Rewriting")
    print("=" * 70)

    # Create the chain
    rag_chain = create_rag_with_rewriting(vectorstore, k=3)

    # Test with a contextual query
    test_query = "What are the benefits?"
    test_history = [
        ("user", "How do I create indexes in MongoDB?"),
        ("assistant", "You can create indexes using createIndex()...")
    ]

    print(f"\n❓ Query: \"{test_query}\"")
    print("   Context: Previous discussion about MongoDB indexes")

    result = rag_chain.invoke({
        "query": test_query,
        "chat_history": test_history
    })

    print(f"\n🔄 Rewritten: \"{result['rewritten_query']}\"")
    print(f"\n✅ Answer:\n{result['answer']}")
    print(f"\n📚 Sources: {len(result['source_chunks'])} chunks retrieved")

    # -------------------------------------------------------------------------
    # Example 4: Pure LCEL Chain
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📌 Example 4: Pure LCEL Implementation")
    print("=" * 70)

    lcel_chain = create_pure_lcel_rewrite_chain(vectorstore, k=2)

    clear_query = "How do I update documents in MongoDB?"
    print(f"\n❓ Query: \"{clear_query}\"")

    lcel_result = lcel_chain.invoke({"query": clear_query})

    print(f"\n🔄 Rewritten: \"{lcel_result['rewritten_query']}\"")
    print(f"\n✅ Answer:\n{lcel_result['answer']}")

    # -------------------------------------------------------------------------
    # Expected Output Format
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📌 Expected Output Format")
    print("=" * 70)
    print("""
{
  "original_query": "How does it work?",
  "rewritten_query": "How does MongoDB's insertOne() method work for adding documents?",
  "answer": "The insertOne() method works by...",
  "sources": [
    {"content": "...", "metadata": {...}}
  ]
}
""")


# =============================================================================
# Summary: Query Rewriting Key Points
# =============================================================================
"""
Query Rewriting for RAG:
------------------------

WHY REWRITE QUERIES?
- Ambiguous queries ("How does it work?") fail in vector search
- Follow-up questions lose context ("What about the other one?")
- User queries often lack searchable keywords
- Pronouns have no meaning to embedding models

TWO-STEP PIPELINE:
1. REWRITE: Transform query into standalone, specific question
2. RETRIEVE: Use rewritten query for vector similarity search

REWRITING STRATEGIES:
1. Simple rewriting (no context):
   - Add keywords, remove ambiguity
   - Good for single-turn interactions

2. Contextual rewriting (with chat history):
   - Resolve pronouns using conversation context
   - Essential for multi-turn conversations

3. HyDE (Hypothetical Document Embeddings):
   - Generate hypothetical answer, embed that
   - Advanced technique for complex queries

LCEL IMPLEMENTATION:
-------------------
rewrite_chain = prompt | model | parser

rag_with_rewrite = (
    RunnablePassthrough.assign(
        rewritten_query=rewrite_chain
    )
    | RunnablePassthrough.assign(
        docs=lambda x: retriever.invoke(x["rewritten_query"])
    )
    | answer_chain
)

PRO TIPS:
---------
- Always log both original and rewritten queries for debugging
- Use chat history for context in conversational RAG
- Compare retrieval scores before/after rewriting
- Consider using multiple rewrites (query expansion)
- Test with intentionally vague queries to validate
"""
