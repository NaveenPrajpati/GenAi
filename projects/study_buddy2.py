#!/usr/bin/env python3
"""
Smart Study Buddy - Q&A Tutor with RAG (Functional Version)
A chatbot that uses retrieval-augmented generation to answer questions
about various topics using Wikipedia and other content sources.
"""

import os
import streamlit as st
from typing import List, Dict, Any
import wikipediaapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# 1. GLOBAL CONFIG (load API key from .env)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PERSIST_DIR = "./chroma_db"

# 2. Create (or get) the Wikipedia API instance
def get_wikipedia_api():
    """
    Returns a Wikipedia API object for English with appropriate settings.
    """
    return wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent='StudyBuddy/1.0 (https://example.com)'
    )

# 3. Create or load vector store
def get_vectorstore(persist_directory: str, embedding_function) -> Chroma:
    """
    Initializes or loads a Chroma vector store with persistence.
    """
    os.makedirs(persist_directory, exist_ok=True)
    return Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_function
    )

# 4. Create the text splitter
def get_text_splitter():
    """
    Returns a text splitter for chunking documents.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )

# 5. Create the OpenAI LLM and embedding function
def get_llm(api_key: str):
    """Returns a ChatOpenAI instance."""
    return ChatOpenAI(
        temperature=0.7,
        model_name="gpt-3.5-turbo"
    )

def get_embeddings(api_key: str):
    """Returns OpenAIEmbeddings instance."""
    return OpenAIEmbeddings()

# 6. Create the RetrievalQA chain
def create_qa_chain(llm, vectorstore):
    """
    Constructs a RetrievalQA chain with a custom prompt.
    """
    prompt_template = """You are a helpful study buddy and tutor. Use the following context to answer the question in a clear, educational manner. 

If you don't know the answer based on the context, say so. Always try to explain concepts in a way that's easy to understand, and provide examples when helpful.

Context: {context}

Question: {question}

Answer: """
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

# 7. Ingest Wikipedia content
def ingest_wikipedia_content(topics: List[str], wiki, text_splitter, vectorstore, qa_chain_callback=None) -> Dict[str, Any]:
    """
    Downloads Wikipedia articles for given topics, chunks them, and adds them to the vector store.
    Optionally, updates the QA chain via the callback after adding new data.
    """
    documents = []
    ingested_topics = []
    failed_topics = []
    for topic in topics:
        try:
            page = wiki.page(topic)
            if page.exists():
                doc = Document(
                    page_content=page.text,
                    metadata={
                        "source": f"Wikipedia: {page.title}",
                        "title": page.title,
                        "url": page.fullurl,
                        "type": "wikipedia"
                    }
                )
                documents.append(doc)
                ingested_topics.append(topic)
            else:
                failed_topics.append(f"{topic} (page not found)")
        except Exception as e:
            failed_topics.append(f"{topic} (error: {str(e)})")
    if documents:
        chunks = text_splitter.split_documents(documents)
        vectorstore.add_documents(chunks)
        vectorstore.persist()
        if qa_chain_callback:
            qa_chain_callback()
        return {
            "ingested": ingested_topics,
            "failed": failed_topics,
            "total_chunks": len(chunks)
        }
    else:
        return {
            "ingested": [],
            "failed": failed_topics,
            "total_chunks": 0
        }

# 8. Ingest custom content
def ingest_custom_content(title: str, content: str, text_splitter, vectorstore, qa_chain_callback=None) -> bool:
    """
    Ingests user-provided content as a document.
    """
    try:
        doc = Document(
            page_content=content,
            metadata={
                "source": f"custom: {title}",
                "title": title,
                "type": "custom"
            }
        )
        chunks = text_splitter.split_documents([doc])
        vectorstore.add_documents(chunks)
        vectorstore.persist()
        if qa_chain_callback:
            qa_chain_callback()
        return True
    except Exception as e:
        st.error(f"Error ingesting custom content: {e}")
        return False

# 9. Answer user questions
def answer_question(question: str, qa_chain) -> Dict[str, Any]:
    """
    Uses the QA chain to answer the user's question.
    """
    try:
        result = qa_chain({"query": question})
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            })
        return {
            "answer": result["result"],
            "sources": sources
        }
    except Exception as e:
        return {
            "answer": f"Error processing question: {str(e)}",
            "sources": []
        }

# 10. Get info about the knowledge base
def get_collection_info(vectorstore) -> Dict[str, Any]:
    """
    Returns info about current Chroma collection (document count, status).
    """
    try:
        collection = vectorstore._collection
        count = collection.count()
        return {
            "document_count": count,
            "status": "Ready" if count > 0 else "Empty"
        }
    except:
        return {
            "document_count": 0,
            "status": "Not initialized"
        }

# 11. Streamlit UI (main function)
def main():
    st.set_page_config(page_title="Smart Study Buddy", page_icon="📚", layout='wide')
    st.title("📚 Smart Study Buddy - Q&A Tutor (Functional)")
    st.markdown("Ask questions about any topic! I'll search through ingested content to provide helpful answers.")

    if not OPENAI_API_KEY:
        st.error("OpenAI API key not found in environment variables.")
        return

    # Initialize all components (will persist for Streamlit session)
    if 'state' not in st.session_state:
        st.session_state.state = {}

        # Initial setup
        st.session_state.state["wiki"] = get_wikipedia_api()
        st.session_state.state["embeddings"] = get_embeddings(OPENAI_API_KEY)
        st.session_state.state["text_splitter"] = get_text_splitter()
        st.session_state.state["vectorstore"] = get_vectorstore(PERSIST_DIR, st.session_state.state["embeddings"])
        st.session_state.state["llm"] = get_llm(OPENAI_API_KEY)

        # Initial QA chain
        def recreate_qa():
            st.session_state.state["qa_chain"] = create_qa_chain(
                st.session_state.state["llm"],
                st.session_state.state["vectorstore"]
            )
        st.session_state.state["recreate_qa"] = recreate_qa
        recreate_qa()

    state = st.session_state.state
    wiki = state["wiki"]
    text_splitter = state["text_splitter"]
    vectorstore = state["vectorstore"]
    qa_chain = state["qa_chain"]
    recreate_qa = state["recreate_qa"]

    # Sidebar
    with st.sidebar:
        st.header("📊 Knowledge Base Status")
        info = get_collection_info(vectorstore)
        st.metric("Documents", info["document_count"])
        st.info(f"Status: {info['status']}")

    # Main content
    col1, col2 = st.columns([1, 1])
    with col1:
        st.header("📖 Content Ingestion")
        st.subheader("Wikipedia Topics")
        wikipedia_topics = st.text_area("Enter Wikipedia topics (one per line):")
        if st.button("📥 Ingest Wikipedia Content"):
            topics = [topic.strip() for topic in wikipedia_topics.split('\n') if topic.strip()]
            if topics:
                with st.spinner(f"Ingesting {len(topics)} topics from Wikipedia..."):
                    result = ingest_wikipedia_content(topics, wiki, text_splitter, vectorstore, recreate_qa)
                if result["ingested"]:
                    st.success(f"✅ Ingested: {', '.join(result['ingested'])}")
                    st.info(f"Created {result['total_chunks']} text chunks")
                if result["failed"]:
                    st.warning(f"⚠️ Failed: {', '.join(result['failed'])}")
            else:
                st.warning("Please enter at least one topic.")

        st.subheader("Custom Text Content")
        custom_title = st.text_input("Content Title:", placeholder="My Study Notes")
        custom_content = st.text_area("Content Text:", placeholder="Enter your custom study material here...")
        if st.button("📥 Ingest Custom Content"):
            if custom_title and custom_content:
                with st.spinner("Ingesting custom content..."):
                    success = ingest_custom_content(custom_title, custom_content, text_splitter, vectorstore, recreate_qa)
                if success:
                    st.success("✅ Custom content ingested successfully!")
                else:
                    st.error("❌ Failed to ingest custom content.")
            else:
                st.warning("Please provide both title and content.")

    with col2:
        st.header("💬 Ask Questions")
        question = st.text_input("Your Question:", key="question_input")
        if st.button("🔍 Ask Question", type="primary"):
            if question:
                with st.spinner("Thinking..."):
                    response = answer_question(question, state["qa_chain"])
                st.subheader("📝 Answer")
                st.write(response["answer"])
                if response["sources"]:
                    st.subheader("📚 Sources")
                    for i, source in enumerate(response["sources"], 1):
                        with st.expander(f"Source {i}: {source['metadata'].get('title', 'Unknown')}"):
                            st.write(f"**Type:** {source['metadata'].get('type', 'Unknown')}")
                            st.write(f"**Source:** {source['metadata'].get('source', 'Unknown')}")
                            st.write(f"**Preview:** {source['content']}")
            else:
                st.warning("Please enter a question.")

if __name__ == "__main__":
    main()