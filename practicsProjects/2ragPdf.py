import os
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

load_dotenv()

# Evaluation imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        faithfulness,
        context_recall,
        context_precision,
    )

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("RAGAS not available. Using custom evaluation metrics only.")

# Semantic chunking (alternative strategy)
from langchain_experimental.text_splitter import SemanticChunker


@dataclass
class EvaluationResult:
    """Results from evaluating the Q&A system"""

    accuracy: float
    avg_response_time: float
    citation_coverage: float
    chunking_strategy: str
    metrics: Dict[str, float]


class ChunkingStrategy:
    """Base class for chunking strategies"""

    def __init__(self, name: str):
        self.name = name

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        raise NotImplementedError


class RecursiveChunker(ChunkingStrategy):
    """Recursive character-based chunking"""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        super().__init__("recursive")
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""],
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


class SemanticChunker(ChunkingStrategy):
    """Semantic-based chunking using embeddings"""

    def __init__(self, embeddings, breakpoint_threshold_type: str = "percentile"):
        super().__init__("semantic")
        self.splitter = SemanticChunker(
            embeddings=embeddings, breakpoint_threshold_type=breakpoint_threshold_type
        )

    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        return self.splitter.split_documents(documents)


class PDFQASystem:
    """Main PDF Q&A system with multiple chunking strategies"""

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        embeddings_model: str = "text-embedding-3-large",
    ):

        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.embeddings = OpenAIEmbeddings(model_name=embeddings_model)
        self.vector_stores = {}
        self.retrievers = {}
        self.qa_chains = {}

        self.chunking_strategies = {
            "recursive": RecursiveChunker(chunk_size=1000, chunk_overlap=200),
            "semantic": SemanticChunker(self.embeddings),
        }

        self.setup_prompt_template()

    def setup_prompt_template(self):
        """Setup the prompt template for Q&A with citations"""
        template = """
        Use the following pieces of context to answer the question at the end. 
        Always include the page number(s) where you found the information in your answer.
        If you don't know the answer, just say that you don't know, don't try to make up an answer.
        
        Format your citations as [Page X] at the end of relevant sentences.
        
        Context:
        {context}
        
        Question: {question}
        
        Answer with citations:
        """

        self.prompt = PromptTemplate(
            template=template, input_variables=["context", "question"]
        )

    def load_pdfs(self, pdf_paths: List[str]) -> Dict[str, List[Document]]:
        """Load PDFs and return documents with page metadata"""
        all_documents = {}

        for pdf_path in pdf_paths:
            print(f"Loading PDF: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Ensure page metadata is preserved
            for i, doc in enumerate(documents):
                if "page" not in doc.metadata:
                    doc.metadata["page"] = i
                doc.metadata["source"] = pdf_path

            all_documents[pdf_path] = documents

        return all_documents

    def create_vector_stores(self, documents_dict: Dict[str, List[Document]]):
        """Create vector stores for each chunking strategy"""
        # Combine all documents
        all_documents = []
        for docs in documents_dict.values():
            all_documents.extend(docs)

        for strategy_name, chunker in self.chunking_strategies.items():
            print(f"Creating vector store with {strategy_name} chunking...")

            # Chunk documents
            chunked_docs = chunker.chunk_documents(all_documents)
            print(f"Created {len(chunked_docs)} chunks with {strategy_name} strategy")

            # Create FAISS vector store
            vector_store = FAISS.from_documents(
                documents=chunked_docs, embedding=self.embeddings
            )

            self.vector_stores[strategy_name] = vector_store

            # Create retriever
            retriever = vector_store.as_retriever(
                search_type="similarity", search_kwargs={"k": 4}
            )
            self.retrievers[strategy_name] = retriever

            # Create QA chain
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever,
                chain_type_kwargs={"prompt": self.prompt},
                return_source_documents=True,
            )
            self.qa_chains[strategy_name] = qa_chain

    def answer_question(
        self, question: str, strategy: str = "recursive"
    ) -> Dict[str, Any]:
        """Answer a question using specified chunking strategy"""
        if strategy not in self.qa_chains:
            raise ValueError(
                f"Strategy '{strategy}' not available. Choose from: {list(self.qa_chains.keys())}"
            )

        result = self.qa_chains[strategy]({"query": question})

        # Extract page numbers from source documents
        pages = set()
        for doc in result["source_documents"]:
            if "page" in doc.metadata:
                pages.add(doc.metadata["page"])

        return {
            "answer": result["result"],
            "source_pages": sorted(list(pages)),
            "source_documents": result["source_documents"],
            "strategy_used": strategy,
        }


# LangGraph workflow for orchestrating Q&A
class QAState(TypedDict):
    """State for the Q&A workflow"""

    question: str
    strategy: str
    answer: str
    sources: List[int]
    confidence: float


def create_qa_workflow(qa_system: PDFQASystem) -> StateGraph:
    """Create LangGraph workflow for Q&A process"""

    def answer_node(state: QAState) -> QAState:
        """Node to generate answer"""
        result = qa_system.answer_question(state["question"], state["strategy"])

        return {**state, "answer": result["answer"], "sources": result["source_pages"]}

    def confidence_node(state: QAState) -> QAState:
        """Node to calculate confidence score"""
        # Simple confidence based on number of sources and answer length
        confidence = min(
            len(state["sources"]) * 0.2 + len(state["answer"]) * 0.001, 1.0
        )

        return {**state, "confidence": confidence}

    # Build workflow
    workflow = StateGraph(QAState)
    workflow.add_node("answer", answer_node)
    workflow.add_node("confidence", confidence_node)

    workflow.set_entry_point("answer")
    workflow.add_edge("answer", "confidence")
    workflow.add_edge("confidence", END)

    return workflow.compile()


class QAEvaluator:
    """Evaluation framework for Q&A system"""

    def __init__(self, qa_system: PDFQASystem):
        self.qa_system = qa_system

    def create_sample_qa_pairs(self) -> List[Dict[str, Any]]:
        """Create sample Q&A pairs for evaluation"""
        # This would typically be loaded from a file or created manually
        sample_pairs = [
            {
                "question": "What is the main topic discussed in the document?",
                "expected_answer": "Sample expected answer",
                "expected_pages": [1, 2],
            },
            {
                "question": "What are the key findings mentioned?",
                "expected_answer": "Sample key findings",
                "expected_pages": [3, 4, 5],
            },
            # Add more pairs as needed
        ]
        return sample_pairs

    def evaluate_exact_match(self, predicted: str, expected: str) -> float:
        """Simple exact match evaluation"""
        return 1.0 if predicted.lower().strip() == expected.lower().strip() else 0.0

    def evaluate_citation_coverage(
        self, predicted_pages: List[int], expected_pages: List[int]
    ) -> float:
        """Evaluate citation coverage"""
        if not expected_pages:
            return 1.0

        predicted_set = set(predicted_pages)
        expected_set = set(expected_pages)

        intersection = predicted_set.intersection(expected_set)
        return len(intersection) / len(expected_set)

    def evaluate_strategy(
        self, strategy: str, qa_pairs: List[Dict[str, Any]]
    ) -> EvaluationResult:
        """Evaluate a chunking strategy"""
        total_accuracy = 0
        total_citation_coverage = 0
        response_times = []

        results = []

        for qa_pair in qa_pairs:
            import time

            start_time = time.time()

            result = self.qa_system.answer_question(qa_pair["question"], strategy)

            end_time = time.time()
            response_times.append(end_time - start_time)

            # Calculate metrics
            accuracy = self.evaluate_exact_match(
                result["answer"], qa_pair["expected_answer"]
            )
            citation_coverage = self.evaluate_citation_coverage(
                result["source_pages"], qa_pair["expected_pages"]
            )

            total_accuracy += accuracy
            total_citation_coverage += citation_coverage

            results.append(
                {
                    "question": qa_pair["question"],
                    "predicted_answer": result["answer"],
                    "expected_answer": qa_pair["expected_answer"],
                    "accuracy": accuracy,
                    "citation_coverage": citation_coverage,
                    "source_pages": result["source_pages"],
                }
            )

        avg_accuracy = total_accuracy / len(qa_pairs)
        avg_citation_coverage = total_citation_coverage / len(qa_pairs)
        avg_response_time = np.mean(response_times)

        return EvaluationResult(
            accuracy=avg_accuracy,
            avg_response_time=avg_response_time,
            citation_coverage=avg_citation_coverage,
            chunking_strategy=strategy,
            metrics={"individual_results": results, "total_questions": len(qa_pairs)},
        )

    def compare_strategies(
        self, qa_pairs: List[Dict[str, Any]]
    ) -> Dict[str, EvaluationResult]:
        """Compare all chunking strategies"""
        results = {}

        for strategy in self.qa_system.chunking_strategies.keys():
            print(f"Evaluating {strategy} strategy...")
            results[strategy] = self.evaluate_strategy(strategy, qa_pairs)

        return results


def main():
    """Main function demonstrating the PDF Q&A system"""

    # Initialize the system
    print("Initializing PDF Q&A System...")
    qa_system = PDFQASystem()

    # Load PDFs (replace with your PDF paths)
    pdf_paths = [
        "/Users/naveen/Desktop/web/genAi/mongtutorial.pdf",
    ]  # Replace with actual paths

    try:
        # Load documents
        documents_dict = qa_system.load_pdfs(pdf_paths)

        # Create vector stores with different chunking strategies
        qa_system.create_vector_stores(documents_dict)

        # Create LangGraph workflow
        workflow = create_qa_workflow(qa_system)

        # Example usage
        question = "What is the main topic of the document?"

        # Test with recursive chunking
        state = {
            "question": question,
            "strategy": "recursive",
            "answer": "",
            "sources": [],
            "confidence": 0.0,
        }

        result = workflow.invoke(state)
        print(f"\nQuestion: {question}")
        print(f"Answer: {result['answer']}")
        print(f"Source Pages: {result['sources']}")
        print(f"Confidence: {result['confidence']:.2f}")

        # Evaluation
        evaluator = QAEvaluator(qa_system)
        qa_pairs = evaluator.create_sample_qa_pairs()

        if qa_pairs:
            comparison_results = evaluator.compare_strategies(qa_pairs)

            print("\n" + "=" * 50)
            print("EVALUATION RESULTS")
            print("=" * 50)

            for strategy, eval_result in comparison_results.items():
                print(f"\n{strategy.upper()} Strategy:")
                print(f"  Accuracy: {eval_result.accuracy:.2%}")
                print(f"  Citation Coverage: {eval_result.citation_coverage:.2%}")
                print(f"  Avg Response Time: {eval_result.avg_response_time:.2f}s")

        # Save vector stores for reuse
        for strategy, vector_store in qa_system.vector_stores.items():
            vector_store.save_local(f"faiss_index_{strategy}")
            print(f"Saved {strategy} vector store to faiss_index_{strategy}")

    except FileNotFoundError as e:
        print(f"PDF files not found: {e}")
        print("Please update the pdf_paths list with valid PDF file paths")

    except Exception as e:
        print(f"Error: {e}")


# Additional utility functions


def load_saved_vector_store(strategy: str, embeddings) -> FAISS:
    """Load a previously saved vector store"""
    return FAISS.load_local(f"faiss_index_{strategy}", embeddings)


def create_evaluation_dataset(pdf_paths: List[str], output_path: str):
    """Helper function to create evaluation dataset"""
    qa_pairs = [
        {
            "question": "What is the main objective of this research?",
            "expected_answer": "To be filled manually after reading the document",
            "expected_pages": [1],
        },
        {
            "question": "What methodology was used in this study?",
            "expected_answer": "To be filled manually after reading the document",
            "expected_pages": [2, 3],
        },
        {
            "question": "What are the key findings or results?",
            "expected_answer": "To be filled manually after reading the document",
            "expected_pages": [4, 5, 6],
        },
        {
            "question": "What are the limitations mentioned in the study?",
            "expected_answer": "To be filled manually after reading the document",
            "expected_pages": [7],
        },
        {
            "question": "What future work is suggested?",
            "expected_answer": "To be filled manually after reading the document",
            "expected_pages": [8],
        },
    ]

    with open(output_path, "w") as f:
        json.dump(qa_pairs, f, indent=2)

    print(f"Template evaluation dataset saved to {output_path}")
    print(
        "Please fill in the expected_answer fields manually after reading your documents"
    )


if __name__ == "__main__":
    main()

"""

USAGE EXAMPLES:

# Basic usage
qa_system = PDFQASystem()
documents = qa_system.load_pdfs(["document.pdf"])
qa_system.create_vector_stores(documents)
result = qa_system.answer_question("What is the main topic?")

# Compare strategies
evaluator = QAEvaluator(qa_system)
qa_pairs = evaluator.create_sample_qa_pairs()
results = evaluator.compare_strategies(qa_pairs)

# Use LangGraph workflow
workflow = create_qa_workflow(qa_system)
result = workflow.invoke({"question": "Your question", "strategy": "recursive"})
"""
