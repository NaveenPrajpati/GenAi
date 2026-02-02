"""
Conditional Chain - Branching Based on Conditions
==================================================

LEARNING OBJECTIVES:
- Route data to different chains based on conditions
- Use RunnableBranch for if/else logic
- Combine classification with conditional routing

CONCEPT:
Conditional chains allow different processing paths based on
the content or classification of input. This is essential for:
- Sentiment-based responses
- Intent routing
- Multi-path workflows

VISUAL REPRESENTATION:
    ┌─────────────────────────────────────────────────────────────┐
    │                   Conditional Chain Flow                     │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │           {"feedback": "This phone is amazing!"}            │
    │                           │                                  │
    │                           ▼                                  │
    │                   ┌──────────────┐                          │
    │                   │  Classifier  │                          │
    │                   │   (LLM +     │                          │
    │                   │   Pydantic)  │                          │
    │                   └──────┬───────┘                          │
    │                          │                                   │
    │                          ▼                                   │
    │                   Feedback(sentiment="positive")            │
    │                          │                                   │
    │          ┌───────────────┼───────────────┐                  │
    │          │ positive?     │ negative?     │ default          │
    │          ▼               ▼               ▼                   │
    │   ┌────────────┐  ┌────────────┐  ┌────────────┐           │
    │   │  Positive  │  │  Negative  │  │  Fallback  │           │
    │   │  Response  │  │  Response  │  │  Handler   │           │
    │   └────────────┘  └────────────┘  └────────────┘           │
    │          │               │               │                   │
    │          └───────────────┴───────────────┘                  │
    │                          │                                   │
    │                          ▼                                   │
    │                    Final Response                            │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

COMPONENTS:
1. Classifier Chain: Determines the category/sentiment
2. RunnableBranch: Routes to appropriate handler
3. Handler Chains: Process based on classification

USE CASES:
- Customer feedback routing
- Support ticket triage
- Content moderation
- Multi-language routing

PREREQUISITES:
- Completed: chains/simpleChain.py, chains/parallelChain.py
- Understanding of Pydantic models
- OpenAI API key in .env

NEXT STEPS:
- runnables/runnableBranch.py - More branching patterns
- practicsProjects/5triageAgent.py - Production triage system
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, PydanticOutputParser
from langchain_core.runnables import RunnableBranch, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# STEP 1: Define the Classification Schema
# =============================================================================
# Pydantic model ensures structured, validated output from LLM


class Feedback(BaseModel):
    """Schema for sentiment classification result."""
    sentiment: Literal["positive", "negative"] = Field(
        description="The sentiment of the feedback: 'positive' or 'negative'"
    )


# Create parser that converts LLM output to Feedback object
classification_parser = PydanticOutputParser(pydantic_object=Feedback)

# =============================================================================
# STEP 2: Create the Classifier Chain
# =============================================================================
# This chain analyzes feedback and outputs structured sentiment

classifier_prompt = PromptTemplate(
    template="""Classify the sentiment of the following customer feedback.

Feedback: {feedback}

{format_instruction}""",
    input_variables=["feedback"],
    partial_variables={
        "format_instruction": classification_parser.get_format_instructions()
    }
)

# Initialize model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# Classifier chain outputs a Feedback object
classifier_chain = classifier_prompt | model | classification_parser

# =============================================================================
# STEP 3: Define Response Handlers
# =============================================================================
# Different prompts for different sentiment types

positive_prompt = PromptTemplate(
    template="""Write a warm, appreciative response to this positive customer feedback.
Thank them for their kind words and encourage them to continue using our service.

Feedback: {feedback}

Response:""",
    input_variables=["feedback"]
)

negative_prompt = PromptTemplate(
    template="""Write an empathetic, helpful response to this negative customer feedback.
Apologize for their experience, acknowledge their concerns, and offer to help resolve the issue.

Feedback: {feedback}

Response:""",
    input_variables=["feedback"]
)

# =============================================================================
# STEP 4: Create Conditional Branch
# =============================================================================
# RunnableBranch takes a list of (condition, chain) tuples
# The first matching condition's chain is executed
# The last item (without condition) is the default fallback


def extract_feedback(classification_result):
    """Extract original feedback from the chain input for response generation."""
    # The classification result is a Feedback object
    # We need to pass it along with original input
    return classification_result


# We need a way to pass the original feedback through
# Using a custom approach with RunnableLambda


def make_branch_chain():
    """Create the conditional branch that routes based on sentiment."""

    branch = RunnableBranch(
        # Condition 1: Positive sentiment
        (
            lambda x: x.sentiment == "positive",
            positive_prompt | model | parser
        ),
        # Condition 2: Negative sentiment
        (
            lambda x: x.sentiment == "negative",
            negative_prompt | model | parser
        ),
        # Default fallback (required)
        RunnableLambda(lambda x: "I couldn't determine the sentiment. "
                                  "Please contact our support team directly.")
    )
    return branch


branch_chain = make_branch_chain()

# =============================================================================
# STEP 5: Create Full Pipeline with Context Preservation
# =============================================================================
# We need to preserve the original feedback for response generation


def create_full_chain():
    """Create a chain that classifies and responds appropriately."""

    def classify_and_respond(inputs: dict) -> str:
        feedback = inputs["feedback"]

        # Step 1: Classify the feedback
        classification = classifier_chain.invoke({"feedback": feedback})

        # Step 2: Route based on classification
        if classification.sentiment == "positive":
            response = (positive_prompt | model | parser).invoke({"feedback": feedback})
        else:
            response = (negative_prompt | model | parser).invoke({"feedback": feedback})

        return response

    return RunnableLambda(classify_and_respond)


# Alternative: Using the simpler branch approach
chain = classifier_chain | branch_chain

# =============================================================================
# STEP 6: Execute and Display
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("CONDITIONAL CHAIN DEMO - Customer Feedback Router")
    print("=" * 70)

    # Test cases
    test_feedbacks = [
        "This is a beautiful phone! Best purchase I've ever made!",
        "Terrible service. The product broke after one day and nobody helped.",
        "The app keeps crashing. Very frustrating experience.",
        "I love how easy it is to use. Great job on the design!"
    ]

    for feedback in test_feedbacks:
        print(f"\n{'=' * 70}")
        print(f"FEEDBACK: {feedback}")
        print("-" * 70)

        # First, show the classification
        classification = classifier_chain.invoke({"feedback": feedback})
        print(f"SENTIMENT: {classification.sentiment.upper()}")
        print("-" * 70)

        # Generate appropriate response
        if classification.sentiment == "positive":
            response = (positive_prompt | model | parser).invoke({"feedback": feedback})
        else:
            response = (negative_prompt | model | parser).invoke({"feedback": feedback})

        print(f"RESPONSE:\n{response}")

    # Visualize chain structure
    print("\n" + "=" * 70)
    print("CHAIN STRUCTURE:")
    print("=" * 70)
    chain.get_graph().print_ascii()

    # ==========================================================================
    # ADVANCED: Multi-category classification
    # ==========================================================================
    print("\n" + "=" * 70)
    print("ADVANCED: Multi-Category Classification")
    print("=" * 70)

    class MultiCategory(BaseModel):
        """Schema for multi-category classification."""
        category: Literal["question", "complaint", "praise", "suggestion"] = Field(
            description="The category of the customer message"
        )
        urgency: Literal["low", "medium", "high"] = Field(
            description="How urgent is this message"
        )

    multi_parser = PydanticOutputParser(pydantic_object=MultiCategory)

    multi_prompt = PromptTemplate(
        template="""Classify the following customer message:

Message: {message}

{format_instruction}""",
        input_variables=["message"],
        partial_variables={"format_instruction": multi_parser.get_format_instructions()}
    )

    multi_classifier = multi_prompt | model | multi_parser

    test_messages = [
        "How do I reset my password?",
        "Your product is garbage!!! I want my money back NOW!!!",
        "Just wanted to say your team is fantastic!",
        "Have you considered adding dark mode?"
    ]

    for msg in test_messages:
        result = multi_classifier.invoke({"message": msg})
        print(f"\nMessage: {msg[:50]}...")
        print(f"  Category: {result.category}, Urgency: {result.urgency}")
