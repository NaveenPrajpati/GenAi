import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class SentimentAnalysis(BaseModel):
    """Structured output for sentiment analysis"""

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="The sentiment classification of the input text"
    )
    confidence: float = Field(
        description="Confidence score between 0 and 1", ge=0.0, le=1.0
    )


def analyze_sentiment_structured(text: str) -> SentimentAnalysis:
    """
    Analyze sentiment with structured output

    Args:
        text (str): Input text to analyze

    Returns:
        SentimentAnalysis: Structured sentiment result
    """
    # Create LLM with structured output
    llm = ChatOpenAI()

    # Use structured output
    structured_llm = llm.with_structured_output(SentimentAnalysis)

    prompt = PromptTemplate.from_template(
        """Analyze the sentiment of the following text.
        
Text: "{text}"

Classify the sentiment as positive, negative, or neutral, and provide a confidence score."""
    )

    chain = prompt | structured_llm

    try:
        result = chain.invoke({"text": text})
        return result
    except Exception as e:
        print(f"Error: {e}")
        # Return default structured response
        return SentimentAnalysis(sentiment="neutral", confidence=0.0)


def main():
    """Main function for structured sentiment analysis"""
    print("Structured Sentiment Analysis Tool")

    print("-" * 40)

    while True:
        user_input = input("\nEnter text to analyze (or 'quit' to exit): ")

        if user_input.lower() in ["quit", "exit", "q"]:
            break

        if not user_input.strip():
            print("Please enter some text.")
            continue

        result = analyze_sentiment_structured(user_input)
        print(f"Sentiment: {result.sentiment}")

        print(f"Confidence: {result.confidence:.2f}")


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY in your environment")

    else:
        main()
