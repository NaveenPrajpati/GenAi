import os
from typing import List, Dict, Any
import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import CharacterTextSplitter

import numpy as np
from dotenv import load_dotenv
from tqdm.auto import tqdm

load_dotenv()


class BookRecommenderSystem:
    def __init__(self):
        # Initialize LangChain components
        self.embeddings = OpenAIEmbeddings()
        self.llm = ChatOpenAI(temperature=0.7, max_tokens=500)
        self.vectorstore = None
        self.books_data = []

        # Initialize the recommendation prompt template
        self.recommendation_prompt = PromptTemplate.from_template(
            """
            Based on the user's preferences and similar books found, provide personalized book recommendations.
            
            User Preferences: {user_preferences}
            
            Similar Books Found:
            {similar_books}
            
            Context: {context}
            
            Please provide 5 detailed book recommendations with explanations of why each book matches the user's preferences. 
            Format your response as:
            
            **Recommendation 1: [Title] by [Author]**
            - Genre: [Genre]
            - Why recommended: [Detailed explanation]
            
            Continue this format for all 5 recommendations.
            """
        )

        self.recommendation_chain = self.recommendation_prompt | self.llm

    def load_books_data(self, books_data: List[Dict[str, Any]]):
        """
        Load books data into the system

        Args:
            books_data: List of dictionaries containing book information
                       Expected keys: title, author, genre, description, rating, year
        """
        self.books_data = books_data

        # Create documents for vector storage
        texts = []
        metadatas = []
        for book in books_data:
            content = (
                f"Title: {book['title']}\n"
                f"Author: {book['author']}\n"
                f"Genre: {book['genre']}\n"
                f"Description: {book['description']}\n"
                f"Rating: {book.get('rating', 'N/A')}\n"
                f"Year: {book.get('year', 'N/A')}\n"
            )
            texts.append(content)
            metadatas.append(
                {
                    "title": book["title"],
                    "author": book["author"],
                    "genre": book["genre"],
                    "rating": book.get("rating", 0),
                    "year": book.get("year", 0),
                }
            )

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        docs = text_splitter.create_documents(texts=texts, metadatas=metadatas)

        # Incrementally add with progress
        self.vectorstore = None
        for doc in tqdm(docs, desc="Ingesting chunks into FAISS"):
            if self.vectorstore:
                self.vectorstore.add_documents([doc])
            else:
                self.vectorstore = FAISS.from_documents([doc], self.embeddings)

        print(f"Indexed {len(docs)} chunks from {len(books_data)} books.")

    def find_similar_books(
        self, query: str, k: int = 10, genre_filter: str = None
    ) -> List[Document]:
        """
        Find similar books based on a query using vector similarity

        Args:
            query: User's preferences or search query
            k: Number of similar books to return

        Returns:
            List of similar book documents
        """
        if not self.vectorstore:
            raise ValueError(
                "No books data loaded. Please call load_books_data() first."
            )
        kwargs = {}
        if genre_filter:
            kwargs["filter"] = {"genre": genre_filter}
        return self.vectorstore.similarity_search(query, k=k, **kwargs)

    def get_recommendations(self, user_preferences: str, context: str = "") -> str:
        """
        Get personalized book recommendations based on user preferences

        Args:
            user_preferences: Description of what the user likes
            context: Additional context (e.g., mood, occasion)

        Returns:
            Formatted recommendations
        """
        # Find similar books based on preferences
        similar_books = self.find_similar_books(user_preferences, k=8)

        # Format similar books for the prompt
        similar_books_text = ""
        for i, book in enumerate(similar_books, 1):
            similar_books_text += f"{i}. {book.metadata['title']} by {book.metadata['author']} - {book.metadata['genre']}\n"

        # Generate recommendations using LLM
        recommendations = self.recommendation_chain.run(
            user_preferences=user_preferences,
            similar_books=similar_books_text,
            context=context if context else "General reading recommendations",
        )

        return recommendations

    def get_genre_recommendations(
        self, genre: str, min_rating: float = 4.0
    ) -> List[Dict]:
        """
        Get top-rated books from a specific genre

        Args:
            genre: Target genre
            min_rating: Minimum rating threshold

        Returns:
            List of recommended books from the genre
        """
        genre_books = [
            book
            for book in self.books_data
            if genre.lower() in book.get("genre", "").lower()
            and book.get("rating", 0) >= min_rating
        ]

        # Sort by rating
        genre_books.sort(key=lambda x: x.get("rating", 0), reverse=True)

        return genre_books[:10]  # Return top 10

    def search_books(self, query: str) -> List[Dict]:
        """
        Search books by title, author, or keywords

        Args:
            query: Search query

        Returns:
            List of matching books
        """
        results = []
        query_lower = query.lower()

        for book in self.books_data:
            if (
                query_lower in book.get("title", "").lower()
                or query_lower in book.get("author", "").lower()
                or query_lower in book.get("description", "").lower()
            ):
                results.append(book)

        return results


# Example usage and demo
def create_sample_books_data():
    """Create sample books data for demonstration"""
    return [
        {
            "title": "Dune",
            "author": "Frank Herbert",
            "genre": "Science Fiction",
            "description": "Epic space opera about politics, religion, and ecology on the desert planet Arrakis",
            "rating": 4.6,
            "year": 1965,
        },
        {
            "title": "The Hobbit",
            "author": "J.R.R. Tolkien",
            "genre": "Fantasy",
            "description": "Adventure story of Bilbo Baggins and his journey to reclaim treasure from a dragon",
            "rating": 4.7,
            "year": 1937,
        },
        {
            "title": "1984",
            "author": "George Orwell",
            "genre": "Dystopian Fiction",
            "description": "Totalitarian society where Big Brother watches everyone and controls thought",
            "rating": 4.5,
            "year": 1949,
        },
        {
            "title": "Pride and Prejudice",
            "author": "Jane Austen",
            "genre": "Romance",
            "description": "Classic romance about Elizabeth Bennet and Mr. Darcy in 19th century England",
            "rating": 4.4,
            "year": 1813,
        },
        {
            "title": "The Girl with the Dragon Tattoo",
            "author": "Stieg Larsson",
            "genre": "Mystery/Thriller",
            "description": "Dark thriller about a journalist and hacker investigating a wealthy family's secrets",
            "rating": 4.3,
            "year": 2005,
        },
        {
            "title": "Sapiens",
            "author": "Yuval Noah Harari",
            "genre": "Non-fiction/History",
            "description": "Brief history of humankind from Stone Age to present day",
            "rating": 4.5,
            "year": 2011,
        },
        {
            "title": "The Martian",
            "author": "Andy Weir",
            "genre": "Science Fiction",
            "description": "Survival story of an astronaut stranded on Mars using science and ingenuity",
            "rating": 4.6,
            "year": 2011,
        },
        {
            "title": "Where the Crawdads Sing",
            "author": "Delia Owens",
            "genre": "Fiction/Mystery",
            "description": "Coming-of-age story set in the marshlands of North Carolina with a murder mystery",
            "rating": 4.4,
            "year": 2018,
        },
    ]


def demo_recommender_system():
    """Demonstrate the book recommender system"""
    print("=== Book Recommender System Demo ===\n")

    # Initialize the system
    recommender = BookRecommenderSystem()

    # Load sample data
    books_data = create_sample_books_data()
    recommender.load_books_data(books_data)

    print("1. Basic Search:")
    ss = input("Search book here - ")
    search_results = recommender.search_books(ss)
    for book in search_results:
        print(f"- {book['title']} by {book['author']}")

    print("\n2. Genre Recommendations:")
    fantasy_books = recommender.get_genre_recommendations("Fantasy", min_rating=4.0)
    for book in fantasy_books:
        print(f"- {book['title']} ({book['rating']}/5)")

    print("\n3. Personalized Recommendations:")
    user_prefs = "I enjoy science fiction with strong world-building and complex characters. I like stories about survival and human ingenuity."

    # Note: This would require a valid API key to work
    # recommendations = recommender.get_recommendations(user_prefs)
    # print(recommendations)

    print("System ready for personalized recommendations!")
    # print("(Requires valid OpenAI API key for LLM-powered recommendations)")


if __name__ == "__main__":
    demo_recommender_system()
