from typing_extensions import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv
import re

load_dotenv()


class ContentState(TypedDict):
    user_input: str
    content_type: str  # "question", "creative", "technical", "complaint"
    sentiment: str  # "positive", "negative", "neutral"
    complexity: str  # "simple", "medium", "complex"
    language: str  # "en", "es", "fr", etc.
    response: str
    needs_review: bool
    error_message: str


def analyze_content(state: ContentState):
    """Analyze the input content to determine type, sentiment, and complexity"""
    user_input = state["user_input"]

    try:
        # Use GPT-3.5 for initial analysis
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1, max_tokens=100)

        analysis_prompt = f"""
        Analyze the following text and provide:
        1. Content type: question, creative, technical, or complaint
        2. Sentiment: positive, negative, or neutral
        3. Complexity: simple, medium, or complex
        4. Language code (e.g., en, es, fr)
        
        Text: "{user_input}"
        
        Respond in this exact format:
        Type: [type]
        Sentiment: [sentiment]
        Complexity: [complexity]
        Language: [language_code]
        """

        messages = [HumanMessage(content=analysis_prompt)]
        response = llm.invoke(messages)
        analysis = response.content

        # Parse the response
        type_match = re.search(r"Type:\s*(\w+)", analysis, re.IGNORECASE)
        sentiment_match = re.search(r"Sentiment:\s*(\w+)", analysis, re.IGNORECASE)
        complexity_match = re.search(r"Complexity:\s*(\w+)", analysis, re.IGNORECASE)
        language_match = re.search(r"Language:\s*(\w+)", analysis, re.IGNORECASE)

        return {
            "content_type": type_match.group(1).lower() if type_match else "question",
            "sentiment": (
                sentiment_match.group(1).lower() if sentiment_match else "neutral"
            ),
            "complexity": (
                complexity_match.group(1).lower() if complexity_match else "medium"
            ),
            "language": language_match.group(1).lower() if language_match else "en",
            "error_message": "",
        }

    except Exception as e:
        return {
            "error_message": f"Analysis failed: {str(e)}",
            "content_type": "question",
            "sentiment": "neutral",
            "complexity": "medium",
            "language": "en",
        }


def handle_question(state: ContentState):
    """Handle question-type content with GPT-4"""
    try:
        # Choose model based on complexity
        model = "gpt-4" if state["complexity"] == "complex" else "gpt-3.5-turbo"
        llm = ChatOpenAI(model=model, temperature=0.3, max_tokens=500)

        system_prompt = (
            "You are a helpful assistant. Provide clear, accurate answers to questions."
        )
        if state["complexity"] == "complex":
            system_prompt += " Break down complex topics into understandable parts."

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["user_input"]),
        ]

        response = llm.invoke(messages)

        return {
            "response": response.content,
            "needs_review": state["complexity"] == "complex",
        }

    except Exception as e:
        return {"error_message": f"Question handling failed: {str(e)}"}


def handle_creative(state: ContentState):
    """Handle creative content requests with GPT-4"""
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0.8, max_tokens=800)

        creative_prompt = (
            f"Create engaging, creative content for: {state['user_input']}"
        )

        messages = [
            SystemMessage(
                content="You are a creative writer. Generate imaginative, engaging content."
            ),
            HumanMessage(content=creative_prompt),
        ]

        response = llm.invoke(messages)

        return {
            "response": response.content,
            "needs_review": True,  # Creative content often needs review
        }

    except Exception as e:
        return {"error_message": f"Creative generation failed: {str(e)}"}


def handle_technical(state: ContentState):
    """Handle technical content with GPT-4 and careful review"""
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0.1, max_tokens=1000)

        tech_prompt = f"""
        Provide a technical response to: {state['user_input']}
        
        Requirements:
        - Be accurate and precise
        - Include relevant technical details
        - Provide examples where helpful
        - Cite best practices
        """

        messages = [
            SystemMessage(
                content="You are a technical expert. Provide accurate, detailed technical information."
            ),
            HumanMessage(content=tech_prompt),
        ]

        response = llm.invoke(messages)

        return {
            "response": response.content,
            "needs_review": True,  # Technical content always needs review
        }

    except Exception as e:
        return {"error_message": f"Technical handling failed: {str(e)}"}


def handle_complaint(state: ContentState):
    """Handle complaints with empathetic, solution-focused approach"""
    try:
        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.4, max_tokens=600)

        complaint_prompt = f"""
        Respond empathetically to this complaint: {state['user_input']}
        
        Guidelines:
        - Acknowledge their concerns
        - Show empathy and understanding
        - Offer constructive solutions where possible
        - Maintain a professional, caring tone
        """

        messages = [
            SystemMessage(
                content="You are a customer service expert. Respond with empathy and provide helpful solutions."
            ),
            HumanMessage(content=complaint_prompt),
        ]

        response = llm.invoke(messages)

        return {
            "response": response.content,
            "needs_review": state["sentiment"]
            == "negative",  # Negative complaints need review
        }

    except Exception as e:
        return {"error_message": f"Complaint handling failed: {str(e)}"}


def review_response(state: ContentState):
    """Review and potentially refine the response"""
    try:
        llm = ChatOpenAI(model="gpt-4", temperature=0.2, max_tokens=800)

        review_prompt = f"""
        Review this response for accuracy, tone, and helpfulness:
        
        Original request: {state['user_input']}
        Response: {state['response']}
        
        Provide an improved version if needed, or confirm it's good as-is.
        Start your response with either "APPROVED:" or "IMPROVED:" followed by the response.
        """

        messages = [
            SystemMessage(
                content="You are a quality reviewer. Ensure responses are accurate, helpful, and appropriate."
            ),
            HumanMessage(content=review_prompt),
        ]

        response = llm.invoke(messages)
        reviewed_content = response.content

        if reviewed_content.startswith("IMPROVED:"):
            return {"response": reviewed_content[9:].strip()}
        else:
            return {}  # Keep original response

    except Exception as e:
        return {"error_message": f"Review failed: {str(e)}"}


def handle_error(state: ContentState):
    """Handle any errors that occurred during processing"""
    return {
        "response": f"I apologize, but I encountered an error while processing your request: {state['error_message']}"
    }


# Conditional routing functions
def route_after_analysis(state: ContentState) -> str:
    """Route based on analysis results or errors"""
    if state.get("error_message"):
        return "error"

    content_type = state["content_type"]
    if content_type == "question":
        return "question"
    elif content_type == "creative":
        return "creative"
    elif content_type == "technical":
        return "technical"
    elif content_type == "complaint":
        return "complaint"
    else:
        return "question"  # Default fallback


def route_after_generation(state: ContentState) -> str:
    """Route based on whether response needs review"""
    if state.get("error_message"):
        return "error"

    if state.get("needs_review", False):
        return "review"
    else:
        return "end"


# Create the conditional workflow
def create_content_workflow():
    """Create and return the compiled workflow"""
    graph = StateGraph(ContentState)

    # Add nodes
    graph.add_node("analyze", analyze_content)
    graph.add_node("question", handle_question)
    graph.add_node("creative", handle_creative)
    graph.add_node("technical", handle_technical)
    graph.add_node("complaint", handle_complaint)
    graph.add_node("review", review_response)
    graph.add_node("error", handle_error)

    # Add edges
    graph.add_edge(START, "analyze")

    # First conditional: Route based on content type
    graph.add_conditional_edges(
        "analyze",
        route_after_analysis,
        {
            "question": "question",
            "creative": "creative",
            "technical": "technical",
            "complaint": "complaint",
            "error": "error",
        },
    )

    # Second conditional: Route based on review needs
    for node in ["question", "creative", "technical", "complaint"]:
        graph.add_conditional_edges(
            node,
            route_after_generation,
            {"review": "review", "end": END, "error": "error"},
        )

    # Terminal edges
    graph.add_edge("review", END)
    graph.add_edge("error", END)

    return graph.compile()


# Example usage
if __name__ == "__main__":
    # Note: You'll need to set your OpenAI API key in environment
    # export OPENAI_API_KEY="your-api-key-here"

    app = create_content_workflow()

    # Example inputs to test different paths
    test_inputs = [
        "How does photosynthesis work in plants?",  # Question path
        "Write a short story about a robot learning to dream",  # Creative path
        "Explain the difference between REST and GraphQL APIs",  # Technical path
        "I'm frustrated that my order was delayed again!",  # Complaint path
    ]

    for user_input in test_inputs:
        print(f"\n{'='*50}")
        print(f"Input: {user_input}")
        print(f"{'='*50}")

        try:
            result = app.invoke({"user_input": user_input})

            print(f"Content Type: {result.get('content_type', 'Unknown')}")
            print(f"Sentiment: {result.get('sentiment', 'Unknown')}")
            print(f"Complexity: {result.get('complexity', 'Unknown')}")
            print(f"Reviewed: {'Yes' if result.get('needs_review') else 'No'}")
            print(f"\nResponse:\n{result['response']}")

        except Exception as e:
            print(f"Error processing request: {str(e)}")
            print("Make sure you have set your OPENAI_API_KEY environment variable")
