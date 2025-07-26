#!/usr/bin/env python3
"""
Smart Study Buddy - Q&A Tutor with RAG
A chatbot that uses retrieval-augmented generation to answer questions
about various topics using Wikipedia and other content sources.
"""

import os
from pymongo import MongoClient
import streamlit as st
from typing import List, Dict, Any
import wikipediaapi
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.schema import Document
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.config import Settings
import requests
from bs4 import BeautifulSoup
import tempfile
import shutil
from langchain_mongodb import MongoDBAtlasVectorSearch
from dotenv import load_dotenv
load_dotenv()

class StudyBuddy:
    """
    Core class to handle content ingestion, embedding, vector storage,
    and retrieval-augmented question answering using OpenAI and Chroma.
    """

    def __init__(self, openai_api_key: str, persist_directory: str = "./chroma_db"):
        """
        Initialize embeddings, LLM, text splitter, Wikipedia, vectorstore and QA chain.
        """
        self.openai_api_key = openai_api_key
        self.persist_directory = persist_directory
        
        # Initialize components
        self.embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
        self.llm = ChatOpenAI(
            temperature=0.7,
            openai_api_key=openai_api_key,
            model_name="gpt-3.5-turbo"
        )
        
        # Text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Initialize Wikipedia API
        self.wiki = wikipediaapi.Wikipedia(
            language='en',
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            user_agent='StudyBuddy/1.0 (https://example.com)'
        )
        
        # Initialize or load vector store
        self.vectorstore = None
        self.qa_chain = None
        self._initialize_vectorstore()
    
    def _initialize_vectorstore(self):
        """Initialize the Chroma vector store."""
        try:
            client = MongoClient(os.environ['MONGO_URI'])
            database = client["genai"]
            collection = database["buddy"]
            # Initialize Chroma with persistence
            self.vectorstore = MongoDBAtlasVectorSearch(
    embedding=self.embeddings,
    collection=collection,
    index_name='vector_index',
    relevance_score_fn="cosine",
)
            
            # Create QA chain
            self._create_qa_chain()
            
        except Exception as e:
            st.error(f"Error initializing vector store: {e}")
    
    def _create_qa_chain(self):
        """
        Construct a RetrievalQA chain that uses the Chroma retriever
        and a custom prompt to answer user questions based on ingested context.
        """
        # Custom prompt template
        prompt_template = """You are a helpful study buddy and tutor. Use the following context to answer the question in a clear, educational manner. 

If you don't know the answer based on the context, say so. Always try to explain concepts in a way that's easy to understand, and provide examples when helpful.

Context: {context}

Question: {question}

Answer: """
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
    
    def ingest_wikipedia_content(self, topics: List[str]) -> Dict[str, Any]:
        """Ingest content from Wikipedia for given topics."""
        documents = []
        ingested_topics = []
        failed_topics = []
        
        for topic in topics:
            try:
                # Get Wikipedia page
                page = self.wiki.page(topic)
                
                if page.exists():
                    # Create document
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
        
        # Split documents into chunks
        if documents:
            chunks = self.text_splitter.split_documents(documents)
            
            # Add to vector store
            ids = [f"doc_{i}_{hash(chunk.page_content)}" for i, chunk in enumerate(chunks)]
            self.vectorstore.add_documents(chunks, ids=ids)
            
            # Recreate QA chain with updated vectorstore
            self._create_qa_chain()
        
        return {
            "ingested": ingested_topics,
            "failed": failed_topics,
            "total_chunks": len(chunks) if documents else 0
        }
    
    def ingest_text_content(self, title: str, content: str, source_type: str = "custom") -> bool:
        """Ingest custom text content."""
        try:
            doc = Document(
                page_content=content,
                metadata={
                    "source": f"{source_type}: {title}",
                    "title": title,
                    "type": source_type
                }
            )
            
            # Split into chunks
            chunks = self.text_splitter.split_documents([doc])
            
            # Add to vector store
            ids = [f"doc_{i}_{hash(chunk.page_content)}" for i, chunk in enumerate(chunks)]
            self.vectorstore.add_documents(chunks, ids=ids)
            
            # Recreate QA chain
            self._create_qa_chain()
            
            return True
            
        except Exception as e:
            st.error(f"Error ingesting text content: {e}")
            return False
    
    def ask_question(self, question: str) -> Dict[str, Any]:
        """
        Executes the RetrievalQA chain on a user query and returns the answer and document sources.
        """
        try:
            if not self.qa_chain:
                return {
                    "answer": "Please ingest some content first before asking questions.",
                    "sources": []
                }
            
            result = self.qa_chain({"query": question})
            
            # Extract source information
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
    
    def get_collection_info(self) -> Dict[str, Any]:
        """
        Returns metadata about the current Chroma collection, like document count and readiness status.
        """
        try:
            collection = self.vectorstore._collection
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

def handle_wikipedia_ingestion(buddy: StudyBuddy, wikipedia_topics: str):
    """Handles ingestion of multiple Wikipedia topics."""
    if wikipedia_topics:
        topics = [topic.strip() for topic in wikipedia_topics.split('\n') if topic.strip()]
        with st.spinner(f"Ingesting {len(topics)} topics from Wikipedia..."):
            result = buddy.ingest_wikipedia_content(topics)

        if result["ingested"]:
            st.success(f"✅ Successfully ingested: {', '.join(result['ingested'])}")
            st.info(f"Created {result['total_chunks']} text chunks")

        if result["failed"]:
            st.warning(f"⚠️ Failed to ingest: {', '.join(result['failed'])}")
    else:
        st.warning("Please enter at least one topic.")

# Streamlit UI
def main():
    st.set_page_config(
        page_title="Smart Study Buddy",
        page_icon="📚",
        layout="wide"
    )
    
    st.title("📚 Smart Study Buddy - Q&A Tutor")
    st.markdown("Ask questions about any topic! I'll search through ingested content to provide helpful answers.")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            st.error("OpenAI API key not found in environment variables.")
            return
        
        # Initialize Study Buddy
        if 'study_buddy' not in st.session_state:
            with st.spinner("Initializing Study Buddy..."):
                st.session_state.study_buddy = StudyBuddy(openai_key)
        
        buddy = st.session_state.study_buddy
        
        # Collection info
        st.header("📊 Knowledge Base Status")
        info = buddy.get_collection_info()
        st.metric("Documents", info["document_count"])
        st.info(f"Status: {info['status']}")
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📖 Content Ingestion")
        
        # Wikipedia content ingestion
        st.subheader("Wikipedia Topics")
        wikipedia_topics = st.text_area(
            "Enter Wikipedia topics (one per line):",
            placeholder="Black holes\nWorld War II\nQuantum physics\nMachine learning",
            height=100
        )
        
        if st.button("📥 Ingest Wikipedia Content"):
            handle_wikipedia_ingestion(buddy, wikipedia_topics)
        
        # Custom text ingestion
        st.subheader("Custom Text Content")
        custom_title = st.text_input("Content Title:", placeholder="My Study Notes")
        custom_content = st.text_area(
            "Content Text:",
            placeholder="Enter your custom study material here...",
            height=150
        )
        
        if st.button("📥 Ingest Custom Content"):
            if custom_title and custom_content:
                with st.spinner("Ingesting custom content..."):
                    success = buddy.ingest_text_content(custom_title, custom_content)
                
                if success:
                    st.success("✅ Custom content ingested successfully!")
                else:
                    st.error("❌ Failed to ingest custom content.")
            else:
                st.warning("Please provide both title and content.")
    
    with col2:
        st.header("💬 Ask Questions")
        
        # Question input
        question = st.text_input(
            "Your Question:",
            placeholder="What are black holes? How did World War II start?",
            key="question_input"
        )
        
        if st.button("🔍 Ask Question", type="primary"):
            if question:
                with st.spinner("Thinking..."):
                    response = buddy.ask_question(question)
                
                # Display answer
                st.subheader("📝 Answer")
                st.write(response["answer"])
                
                # Display sources
                if response["sources"]:
                    st.subheader("📚 Sources")
                    for i, source in enumerate(response["sources"], 1):
                        with st.expander(f"Source {i}: {source['metadata'].get('title', 'Unknown')}"):
                            st.write(f"**Type:** {source['metadata'].get('type', 'Unknown')}")
                            st.write(f"**Source:** {source['metadata'].get('source', 'Unknown')}")
                            st.write(f"**Preview:** {source['content']}")
            else:
                st.warning("Please enter a question.")
        
        # Chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
        
        # Display recent questions
        if st.session_state.chat_history:
            st.subheader("🕒 Recent Questions")
            for i, (q, a) in enumerate(reversed(st.session_state.chat_history[-3:])):
                with st.expander(f"Q: {q[:50]}..."):
                    st.write(f"**Answer:** {a[:200]}...")

# Instructions for setup
def show_setup_instructions():
    st.markdown("""
    ## 🚀 Setup Instructions
    
    ### Prerequisites
    ```bash
    pip install streamlit langchain openai chromadb wikipedia-api beautifulsoup4 tiktoken
    ```
    
    ### Required Environment Variables
    - `OPENAI_API_KEY`: Your OpenAI API key
    
    ### How to Run
    ```bash
    streamlit run study_buddy.py
    ```
    
    ### Features
    - **Wikipedia Integration**: Automatically fetch and ingest Wikipedia articles
    - **Custom Content**: Add your own study materials
    - **Vector Search**: Semantic similarity search for relevant context
    - **Source Attribution**: See which documents were used to answer questions
    - **Persistent Storage**: Knowledge base persists between sessions
    
    ### Usage Tips
    - Start by ingesting content on topics you want to study
    - Ask specific questions for better answers
    - Use the sources to verify and learn more
    - Add your own notes and study materials for personalized learning
    """)

if __name__ == "__main__":
    main()