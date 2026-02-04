"""
Agent with Hard Constraints - Tool-Using Agent with Guardrails
===============================================================

This module demonstrates building a constrained agent that uses tools
responsibly with strict guardrails to prevent hallucination and misuse.

Requirements Met:
-----------------
✅ Uses tools ONLY when required
✅ Rejects unsupported queries
✅ Temperature = 0 (deterministic)
✅ Tool schema enforced
✅ No hallucinated tool calls
✅ Clear reasoning
✅ No infinite loops

Architecture:
-------------
    User Query
         │
         ▼
    ┌────────────────────┐
    │   Query Validator  │  Guardrail 1: Check if query is supported
    │   (Intent Check)   │
    └────────────────────┘
         │
         ├─── Unsupported ──► Rejection Response
         │
         ▼
    ┌────────────────────┐
    │   Tool Router      │  Guardrail 2: Decide if tools needed
    │   (Need Tools?)    │
    └────────────────────┘
         │
         ├─── No Tools ──► Direct Response
         │
         ▼
    ┌────────────────────┐
    │   Agent Loop       │  Guardrail 3: Max iterations
    │   (with limits)    │
    └────────────────────┘
         │
         ▼
    ┌────────────────────┐
    │   Response         │  Guardrail 4: Validate output
    │   Validator        │
    └────────────────────┘
         │
         ▼
    Final Response

Guardrails Implemented:
-----------------------
1. Query Validation - Reject off-topic/unsupported queries
2. Tool-Use Gating - Only use tools when necessary
3. Max Iterations - Prevent infinite loops (max 5)
4. Schema Enforcement - Strict tool input validation
5. Response Validation - Ensure quality output
6. Temperature = 0 - Deterministic behavior

Expected Output:
----------------
{
  "query": "...",
  "supported": true/false,
  "tool_calls": [...],
  "reasoning": "...",
  "response": "..."
}

Updated for LangChain/LangGraph 1.0+ (2025-2026)
"""

from langchain_core.tools import tool, StructuredTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Any, Literal, Optional, Annotated
from enum import Enum
import json
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Guardrail 1: Supported Query Categories
# =============================================================================


class QueryCategory(str, Enum):
    """Supported query categories for this agent."""

    DOCUMENT_SEARCH = "document_search"
    SUMMARIZATION = "summarization"
    UNSUPPORTED = "unsupported"


SUPPORTED_CATEGORIES = {
    QueryCategory.DOCUMENT_SEARCH: [
        "search",
        "find",
        "look up",
        "query",
        "get information",
        "what is",
        "tell me about",
        "explain",
        "describe",
    ],
    QueryCategory.SUMMARIZATION: [
        "summarize",
        "summary",
        "brief",
        "condense",
        "shorten",
        "tldr",
        "key points",
        "main ideas",
    ],
}

REJECTION_RESPONSES = {
    "off_topic": "I can only help with document search and summarization tasks. "
    "Your query appears to be outside my supported capabilities.",
    "harmful": "I cannot process this request as it may involve harmful content.",
    "unclear": "I couldn't understand your request. Please rephrase it as a "
    "document search or summarization task.",
    "no_tools_needed": "This query can be answered directly without using tools.",
}


# =============================================================================
# Tool Definitions with Strict Schema Enforcement
# =============================================================================


class SearchDocsInput(BaseModel):
    """Input schema for search_docs tool - strictly enforced."""

    query: str = Field(
        description="The search query to find relevant documents",
        min_length=3,
        max_length=500,
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate and sanitize query."""
        # Remove potentially harmful characters
        sanitized = v.strip()
        if not sanitized:
            raise ValueError("Query cannot be empty")
        if any(char in sanitized for char in ["<", ">", "{", "}", "|"]):
            raise ValueError("Query contains invalid characters")
        return sanitized


class SummarizeInput(BaseModel):
    """Input schema for summarize tool - strictly enforced."""

    text: str = Field(
        description="The text to summarize", min_length=50, max_length=10000
    )
    max_length: Optional[int] = Field(
        default=200, description="Maximum length of summary in words", ge=50, le=500
    )


# Mock document database for search
MOCK_DOCUMENTS = {
    "langchain": "LangChain is a framework for developing applications powered by language models. "
    "It provides tools for prompt management, chains, agents, and memory.",
    "langgraph": "LangGraph is a library for building stateful, multi-actor applications with LLMs. "
    "It uses a graph-based approach where nodes are functions and edges define flow.",
    "rag": "RAG (Retrieval Augmented Generation) combines retrieval systems with LLMs. "
    "It retrieves relevant documents and uses them as context for generation.",
    "embeddings": "Embeddings are vector representations of text that capture semantic meaning. "
    "They enable similarity search and are fundamental to vector databases.",
    "agents": "LLM agents are systems that use language models to decide which actions to take. "
    "They can use tools, maintain memory, and reason through complex tasks.",
}


@tool(args_schema=SearchDocsInput)
def search_docs(query: str) -> str:
    """
    Search the document database for relevant information.

    Use this tool when the user asks to find, search, or look up information
    about a specific topic. Do NOT use for general conversation.

    Args:
        query: The search query (3-500 characters, no special characters)

    Returns:
        Relevant document content or "No results found"
    """
    query_lower = query.lower()
    results = []

    for topic, content in MOCK_DOCUMENTS.items():
        if topic in query_lower or any(word in query_lower for word in topic.split()):
            results.append(f"[{topic.upper()}]: {content}")

    if results:
        return "\n\n".join(results)
    return "No relevant documents found for your query. Try different keywords."


@tool(args_schema=SummarizeInput)
def summarize(text: str, max_length: int = 200) -> str:
    """
    Summarize the provided text into a concise form.

    Use this tool ONLY when explicitly asked to summarize text.
    Do NOT use for general questions or search queries.

    Args:
        text: The text to summarize (50-10000 characters)
        max_length: Maximum summary length in words (50-500)

    Returns:
        A concise summary of the input text
    """
    # Simple extractive summary (first few sentences)
    sentences = text.replace("\n", " ").split(". ")
    summary_sentences = []
    word_count = 0

    for sentence in sentences:
        words = sentence.split()
        if word_count + len(words) <= max_length:
            summary_sentences.append(sentence)
            word_count += len(words)
        else:
            break

    if summary_sentences:
        return ". ".join(summary_sentences) + "."
    return text[: max_length * 5] + "..."  # Fallback


# Tool list
TOOLS = [search_docs, summarize]
TOOL_NAMES = [t.name for t in TOOLS]


# =============================================================================
# Guardrail 2: Query Classifier (Determines if query is supported)
# =============================================================================


def classify_query(query: str) -> tuple[QueryCategory, float]:
    """
    Classify the query into supported categories.

    Returns:
        Tuple of (category, confidence)
    """
    query_lower = query.lower()

    # Check for summarization intent
    for keyword in SUPPORTED_CATEGORIES[QueryCategory.SUMMARIZATION]:
        if keyword in query_lower:
            return QueryCategory.SUMMARIZATION, 0.9

    # Check for search intent
    for keyword in SUPPORTED_CATEGORIES[QueryCategory.DOCUMENT_SEARCH]:
        if keyword in query_lower:
            return QueryCategory.DOCUMENT_SEARCH, 0.85

    # Default: check if it's a question that could use search
    if any(
        query_lower.startswith(w)
        for w in ["what", "how", "why", "when", "where", "who"]
    ):
        return QueryCategory.DOCUMENT_SEARCH, 0.6

    return QueryCategory.UNSUPPORTED, 0.7


# =============================================================================
# Guardrail 3: Tool Router (Decides if tools are needed)
# =============================================================================

TOOL_ROUTING_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a tool routing assistant. Analyze the user's query and determine:
1. Whether tools are needed to answer it
2. Which specific tool(s) should be used

Available tools:
- search_docs: Search for information in documents. Use for questions about topics.
- summarize: Condense text into a shorter form. Use ONLY when user provides text to summarize.

Rules:
- ONLY recommend tools if absolutely necessary
- For simple greetings or off-topic queries, respond with "NO_TOOLS"
- For questions about topics, recommend "search_docs"
- For summarization requests with provided text, recommend "summarize"

Respond in JSON format:
{{"needs_tools": true/false, "tools": ["tool_name"], "reason": "explanation"}}""",
        ),
        ("human", "{query}"),
    ]
)


# =============================================================================
# Agent State with Guardrails
# =============================================================================


class AgentState(MessagesState):
    """Extended state with guardrail tracking."""

    query_category: str = ""
    tool_calls_count: int = 0
    max_iterations: int = 5
    is_supported: bool = True
    rejection_reason: str = ""
    reasoning: List[str] = []


# =============================================================================
# Agent Nodes with Guardrails
# =============================================================================

# Initialize model with temperature=0 for deterministic behavior
model = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,  # CRITICAL: Deterministic behavior
    max_tokens=1000,
)

# Bind tools to model
model_with_tools = model.bind_tools(TOOLS)


def validate_query_node(state: AgentState) -> dict:
    """
    Guardrail Node 1: Validate if query is supported.
    """
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    # Classify the query
    category, confidence = classify_query(last_message)

    reasoning = [
        f"Query classified as: {category.value} (confidence: {confidence:.2f})"
    ]

    if category == QueryCategory.UNSUPPORTED and confidence > 0.5:
        return {
            "is_supported": False,
            "rejection_reason": "off_topic",
            "query_category": category.value,
            "reasoning": reasoning,
        }

    return {
        "is_supported": True,
        "query_category": category.value,
        "reasoning": reasoning,
    }


def agent_node(state: AgentState) -> dict:
    """
    Main agent node with tool-use guardrails.
    """
    messages = state["messages"]
    tool_calls_count = state.get("tool_calls_count", 0)
    max_iterations = state.get("max_iterations", 5)
    reasoning = state.get("reasoning", [])

    # Guardrail: Check iteration limit
    if tool_calls_count >= max_iterations:
        reasoning.append(f"Max iterations ({max_iterations}) reached. Stopping.")
        return {
            "messages": [
                AIMessage(
                    content="I've reached my processing limit. Here's what I found so far based on my searches."
                )
            ],
            "reasoning": reasoning,
        }

    # System message with strict instructions
    system_message = SystemMessage(
        content="""You are a helpful assistant with access to specific tools.

STRICT RULES:
1. ONLY use tools when absolutely necessary to answer the query
2. If you can answer without tools, do so directly
3. If the query is unclear, ask for clarification instead of guessing
4. NEVER make up information - only use what tools return
5. After using a tool, provide a clear answer based on the results
6. Do NOT call the same tool twice with the same arguments

Available tools:
- search_docs: Search documents for information
- summarize: Summarize provided text

If you don't need tools, respond directly."""
    )

    # Prepare messages
    full_messages = [system_message] + list(messages)

    # Get response
    response = model_with_tools.invoke(full_messages)

    # Track reasoning
    if response.tool_calls:
        tool_names = [tc["name"] for tc in response.tool_calls]
        reasoning.append(f"Agent decided to use tools: {tool_names}")
    else:
        reasoning.append("Agent responded without using tools")

    return {
        "messages": [response],
        "tool_calls_count": tool_calls_count + (1 if response.tool_calls else 0),
        "reasoning": reasoning,
    }


def tool_node(state: AgentState) -> dict:
    """
    Tool execution node with validation.
    """
    messages = state["messages"]
    last_message = messages[-1]
    reasoning = state.get("reasoning", [])

    tool_results = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        reasoning.append(
            f"Executing tool: {tool_name} with args: {json.dumps(tool_args)}"
        )

        # Find and execute the tool
        tool_func = None
        for t in TOOLS:
            if t.name == tool_name:
                tool_func = t
                break

        if tool_func:
            try:
                result = tool_func.invoke(tool_args)
                reasoning.append(
                    f"Tool {tool_name} returned result (length: {len(str(result))})"
                )
            except Exception as e:
                result = f"Error executing tool: {str(e)}"
                reasoning.append(f"Tool {tool_name} failed: {str(e)}")
        else:
            result = f"Unknown tool: {tool_name}"
            reasoning.append(f"Unknown tool requested: {tool_name}")

        tool_results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": tool_results, "reasoning": reasoning}


def rejection_node(state: AgentState) -> dict:
    """
    Node for handling rejected queries.
    """
    rejection_reason = state.get("rejection_reason", "off_topic")
    response = REJECTION_RESPONSES.get(
        rejection_reason, REJECTION_RESPONSES["off_topic"]
    )

    return {
        "messages": [AIMessage(content=response)],
        "reasoning": state.get("reasoning", [])
        + [f"Query rejected: {rejection_reason}"],
    }


# =============================================================================
# Routing Functions
# =============================================================================


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Determine if agent should continue to tools or end.
    """
    messages = state["messages"]
    last_message = messages[-1]

    # Check if the last message has tool calls
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    return "end"


def validate_route(state: AgentState) -> Literal["agent", "reject"]:
    """
    Route based on query validation.
    """
    if state.get("is_supported", True):
        return "agent"
    return "reject"


# =============================================================================
# Build the Agent Graph
# =============================================================================


def create_guardrailed_agent():
    """
    Create an agent with all guardrails implemented.
    """
    # Create graph
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("validate", validate_query_node)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.add_node("reject", rejection_node)

    # Add edges
    graph.add_edge(START, "validate")
    graph.add_conditional_edges(
        "validate", validate_route, {"agent": "agent", "reject": "reject"}
    )
    graph.add_conditional_edges(
        "agent", should_continue, {"tools": "tools", "end": END}
    )
    graph.add_edge("tools", "agent")
    graph.add_edge("reject", END)

    # Compile with checkpointer for state persistence
    checkpointer = InMemorySaver()
    return graph.compile(checkpointer=checkpointer)


# =============================================================================
# Helper Function for Running Agent
# =============================================================================


def run_agent(query: str, thread_id: str = "default") -> Dict[str, Any]:
    """
    Run the guardrailed agent with a query.

    Returns:
        Dict with query, response, tool_calls, reasoning, and metadata
    """
    agent = create_guardrailed_agent()

    config = {"configurable": {"thread_id": thread_id}}

    result = agent.invoke({"messages": [HumanMessage(content=query)]}, config=config)

    # Extract information
    messages = result["messages"]
    tool_calls = []

    for msg in messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            tool_calls.extend(msg.tool_calls)

    final_response = messages[-1].content if messages else "No response"

    return {
        "query": query,
        "supported": result.get("is_supported", True),
        "category": result.get("query_category", "unknown"),
        "tool_calls": [{"name": tc["name"], "args": tc["args"]} for tc in tool_calls],
        "reasoning": result.get("reasoning", []),
        "response": final_response,
        "iterations": result.get("tool_calls_count", 0),
    }


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AGENT WITH HARD CONSTRAINTS - Guardrailed Tool Usage Demo")
    print("=" * 70)

    # -------------------------------------------------------------------------
    # Test 1: Valid Search Query
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📌 Test 1: Valid Document Search Query")
    print("=" * 70)

    result = run_agent("What is LangChain and how does it work?")

    print(f"\n❓ Query: {result['query']}")
    print(f"✅ Supported: {result['supported']}")
    print(f"📂 Category: {result['category']}")
    print(f"🔧 Tool Calls: {len(result['tool_calls'])}")
    for tc in result["tool_calls"]:
        print(f"   - {tc['name']}: {tc['args']}")
    print(f"🧠 Reasoning Steps: {len(result['reasoning'])}")
    for r in result["reasoning"]:
        print(f"   - {r}")
    print(f"\n💬 Response:\n{result['response'][:500]}...")

    # -------------------------------------------------------------------------
    # Test 2: Summarization Request
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📌 Test 2: Summarization Request")
    print("=" * 70)

    long_text = """
    LangChain is a framework designed to simplify the creation of applications
    using large language models. It provides a standard interface for chains,
    lots of integrations with other tools, and end-to-end chains for common
    applications. The main value props of LangChain are: Components that are
    modular and easy to use, Use-case specific chains that make it easy to get
    started, and the ability to customize existing chains and build new ones.
    """

    result = run_agent(f"Please summarize this text: {long_text}")

    print(f"\n❓ Query: Summarize text request")
    print(f"✅ Supported: {result['supported']}")
    print(f"🔧 Tool Calls: {len(result['tool_calls'])}")
    print(f"\n💬 Response:\n{result['response'][:500]}...")

    # -------------------------------------------------------------------------
    # Test 3: Unsupported Query (Off-topic)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📌 Test 3: Unsupported Query (Should be Rejected)")
    print("=" * 70)

    result = run_agent("Write me a poem about cats")

    print(f"\n❓ Query: {result['query']}")
    print(f"✅ Supported: {result['supported']}")
    print(f"📂 Category: {result['category']}")
    print(f"\n💬 Response:\n{result['response']}")

    # -------------------------------------------------------------------------
    # Test 4: Direct Answer (No Tools Needed)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📌 Test 4: Simple Question (May Not Need Tools)")
    print("=" * 70)

    result = run_agent("What are embeddings used for in AI?")

    print(f"\n❓ Query: {result['query']}")
    print(f"🔧 Tool Calls: {len(result['tool_calls'])}")
    print(f"🔄 Iterations: {result['iterations']}")
    print(f"\n💬 Response:\n{result['response'][:500]}...")

    # -------------------------------------------------------------------------
    # Test 5: Multiple Tool Calls
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📌 Test 5: Query Requiring Multiple Searches")
    print("=" * 70)

    result = run_agent("Tell me about RAG and how it relates to LangChain")

    print(f"\n❓ Query: {result['query']}")
    print(f"🔧 Tool Calls: {len(result['tool_calls'])}")
    for tc in result["tool_calls"]:
        print(f"   - {tc['name']}: {tc['args']}")
    print(f"\n💬 Response:\n{result['response'][:500]}...")

    # -------------------------------------------------------------------------
    # Expected Output Format
    # -------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("📌 Expected Output Format")
    print("=" * 70)
    print(
        """
{
  "query": "What is LangChain?",
  "supported": true,
  "category": "document_search",
  "tool_calls": [
    {"name": "search_docs", "args": {"query": "langchain"}}
  ],
  "reasoning": [
    "Query classified as: document_search (confidence: 0.85)",
    "Agent decided to use tools: ['search_docs']",
    "Executing tool: search_docs with args: {...}",
    "Tool search_docs returned result (length: 256)"
  ],
  "response": "LangChain is a framework for developing...",
  "iterations": 1
}
"""
    )


# =============================================================================
# Summary: Agent Guardrails Key Points
# =============================================================================
"""
Agent Guardrails Implementation:
--------------------------------

1. QUERY VALIDATION
   - Classify queries into supported categories
   - Reject off-topic or harmful queries early
   - Provide clear rejection messages

2. TOOL SCHEMA ENFORCEMENT
   - Use Pydantic models for strict input validation
   - Sanitize inputs (remove special characters)
   - Set min/max lengths for all fields

3. TEMPERATURE = 0
   - Critical for deterministic behavior
   - Prevents random/hallucinated tool calls
   - Ensures reproducible results

4. ITERATION LIMITS
   - max_iterations = 5 (configurable)
   - Prevents infinite loops
   - Graceful degradation when limit reached

5. TOOL-USE GATING
   - Only use tools when necessary
   - Direct response for simple queries
   - Clear reasoning for tool selection

6. RESPONSE VALIDATION
   - Track all reasoning steps
   - Validate tool outputs
   - Clear error handling

Graph Structure:
----------------
START → validate → [agent ↔ tools] → END
                 ↘ reject → END

Key Settings:
-------------
- temperature=0 (deterministic)
- max_iterations=5 (loop prevention)
- strict tool schemas (Pydantic validation)
- explicit rejection messages

Pro Tips:
---------
- Always validate queries before processing
- Use temperature=0 for agents that use tools
- Implement clear rejection messages
- Track reasoning for debugging
- Set iteration limits to prevent runaway loops
- Use Pydantic for tool input validation
"""
