from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma, FAISS
import os
from langchain.chains import RetrievalQA
from langchain.llms import Ollama  # or OpenAI, etc.
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter


def load_pdfs(pdf_paths):
    """Load and process PDF documents"""
    documents = []

    for pdf_path in pdf_paths:
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Add source metadata
        for doc in docs:
            doc.metadata["source_file"] = os.path.basename(pdf_path)
            doc.metadata["page"] = doc.metadata.get("page", 0) + 1  # 1-indexed

        documents.extend(docs)

    return documents


def chunk_strategy_1(documents, chunk_size=1000, chunk_overlap=200):
    """Conservative chunking with overlap"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def chunk_strategy_2(documents, chunk_size=1500, chunk_overlap=100):
    """Larger chunks with semantic boundaries"""
    text_splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n\n",  # Split on paragraphs
    )

    chunks = text_splitter.split_documents(documents)
    return chunks


def create_vector_store(chunks, store_type="chroma", persist_directory="./chroma_db"):
    """Create and populate vector store"""

    # Initialize embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if store_type == "chroma":
        vectorstore = Chroma.from_documents(
            documents=chunks, embedding=embeddings, persist_directory=persist_directory
        )
        vectorstore.persist()

    elif store_type == "faiss":
        vectorstore = FAISS.from_documents(documents=chunks, embedding=embeddings)
        vectorstore.save_local("./faiss_index")

    return vectorstore


def create_rag_chain(vectorstore, llm_model="llama2"):
    """Create RAG chain with citation prompt"""

    # Initialize LLM
    llm = Ollama(model=llm_model)

    # Custom prompt for citations
    prompt_template = """
    Use the following pieces of context to answer the question at the end. 
    Always include the source page number(s) in your answer using the format [Page X].
    If you don't know the answer, say that you don't know.
    
    Context:
    {context}
    
    Question: {question}
    
    Answer with page citations:
    """

    prompt = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    # Create retrieval chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True,
    )

    return qa_chain


def enhance_answer_with_citations(result):
    """Add detailed citations to answers"""
    answer = result["result"]
    source_docs = result["source_documents"]

    # Extract page numbers from source documents
    pages = set()
    sources = []

    for doc in source_docs:
        page_num = doc.metadata.get("page", "Unknown")
        source_file = doc.metadata.get("source_file", "Unknown")
        pages.add(page_num)
        sources.append(f"{source_file} (Page {page_num})")

    # Format citations
    if pages:
        page_list = sorted(list(pages))
        citation = f" [Pages: {', '.join(map(str, page_list))}]"
        if not any(cite in answer for cite in ["[Page", "[Pages"]):
            answer += citation

    return {"answer": answer, "sources": sources, "source_documents": source_docs}


# test_qa_pairs.json
test_questions = [
    {
        "question": "What is the main conclusion of the study?",
        "expected_answer": "The study concludes that...",
        "expected_pages": [12, 15],
    },
    # Add 19 more Q/A pairs
]


def evaluate_rag_system(qa_chain, test_questions):
    """Evaluate RAG system performance"""
    results = {"accuracy": 0, "citation_coverage": 0, "detailed_results": []}

    correct_answers = 0
    cited_answers = 0

    for item in test_questions:
        question = item["question"]
        expected = item["expected_answer"]
        expected_pages = item.get("expected_pages", [])

        # Get RAG answer
        result = qa_chain({"query": question})
        enhanced_result = enhance_answer_with_citations(result)
        answer = enhanced_result["answer"]

        # Check accuracy (simple similarity or exact match)
        is_correct = evaluate_answer_quality(answer, expected)
        if is_correct:
            correct_answers += 1

        # Check citation presence
        has_citation = any(page in answer for page in ["[Page", "[Pages"])
        if has_citation:
            cited_answers += 1

        results["detailed_results"].append(
            {
                "question": question,
                "answer": answer,
                "correct": is_correct,
                "has_citation": has_citation,
            }
        )

    results["accuracy"] = correct_answers / len(test_questions)
    results["citation_coverage"] = cited_answers / len(test_questions)

    return results


def evaluate_answer_quality(answer, expected):
    """Simple evaluation - can be enhanced with semantic similarity"""
    # Basic keyword overlap or use sentence transformers for semantic similarity
    answer_lower = answer.lower()
    expected_lower = expected.lower()

    # Simple keyword matching (enhance as needed)
    common_words = set(answer_lower.split()) & set(expected_lower.split())
    return len(common_words) / len(set(expected_lower.split())) > 0.3


def compare_chunking_strategies(pdf_paths, test_questions):
    """Compare two chunking strategies"""
    documents = load_pdfs(pdf_paths)

    # Strategy 1
    chunks_1 = chunk_strategy_1(documents)
    vectorstore_1 = create_vector_store(chunks_1, persist_directory="./chroma_db_1")
    qa_chain_1 = create_rag_chain(vectorstore_1)
    results_1 = evaluate_rag_system(qa_chain_1, test_questions)

    # Strategy 2
    chunks_2 = chunk_strategy_2(documents)
    vectorstore_2 = create_vector_store(chunks_2, persist_directory="./chroma_db_2")
    qa_chain_2 = create_rag_chain(vectorstore_2)
    results_2 = evaluate_rag_system(qa_chain_2, test_questions)

    # Compare results
    comparison = {
        "strategy_1": {
            "chunks_count": len(chunks_1),
            "accuracy": results_1["accuracy"],
            "citation_coverage": results_1["citation_coverage"],
        },
        "strategy_2": {
            "chunks_count": len(chunks_2),
            "accuracy": results_2["accuracy"],
            "citation_coverage": results_2["citation_coverage"],
        },
    }

    return comparison


def main():
    # Configuration
    pdf_paths = ["document1.pdf", "document2.pdf"]

    # Load and process documents
    documents = load_pdfs(pdf_paths)
    chunks = chunk_strategy_1(documents)  # or strategy_2

    # Create vector store
    vectorstore = create_vector_store(chunks)

    # Create RAG chain
    qa_chain = create_rag_chain(vectorstore)

    # Interactive querying
    while True:
        question = input("Enter your question (or 'quit' to exit): ")
        if question.lower() == "quit":
            break

        result = qa_chain({"query": question})
        enhanced_result = enhance_answer_with_citations(result)

        print(f"Answer: {enhanced_result['answer']}")
        print(f"Sources: {', '.join(enhanced_result['sources'])}")
        print("-" * 50)


if __name__ == "__main__":
    main()
