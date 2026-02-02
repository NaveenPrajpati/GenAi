"""
LangGraph Streaming - Real-Time Output
=======================================

LEARNING OBJECTIVES:
- Stream state updates as the graph executes
- Stream individual node outputs
- Stream token-by-token from LLM calls
- Build responsive UIs with streaming

CONCEPT:
Streaming allows you to get partial results as the graph executes,
rather than waiting for the entire workflow to complete. This is
essential for:
- Responsive chat interfaces
- Progress indicators
- Long-running workflows
- Real-time dashboards

STREAMING MODES:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Streaming Modes                           │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  1. "values" mode:                                          │
    │     - Streams the full state after each node                │
    │     - Good for seeing complete state snapshots              │
    │                                                              │
    │  2. "updates" mode:                                         │
    │     - Streams only the changes made by each node            │
    │     - More efficient for large states                       │
    │                                                              │
    │  3. "messages" mode:                                        │
    │     - Streams individual message tokens                     │
    │     - Best for chat interfaces                              │
    │                                                              │
    │  4. Token streaming:                                        │
    │     - Stream individual tokens from LLM                     │
    │     - Use astream_events() for fine-grained control         │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

VISUAL FLOW:
    User Input → [Node1] → [Node2] → [Node3] → Output
                    ↓          ↓          ↓
                 Stream     Stream     Stream
                 Update     Update     Update

PREREQUISITES:
- Completed: langgraph/first.ipynb
- Understanding of StateGraph basics
- OpenAI API key in .env

NEXT STEPS:
- langgraph/humanInTheLoop.py - Interrupt and resume
- langgraph/persistance.py - State checkpointing
"""

from typing import Annotated, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
import asyncio

load_dotenv()

# =============================================================================
# STEP 1: Define State
# =============================================================================


class ChatState(TypedDict):
    """State for our streaming chat example."""
    messages: Annotated[list[BaseMessage], add_messages]
    current_step: str


# =============================================================================
# STEP 2: Create Nodes
# =============================================================================

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7, streaming=True)


def analyze_input(state: ChatState) -> ChatState:
    """Analyze the user's input."""
    print("  [analyze_input] Analyzing user message...")
    return {"current_step": "analyze"}


def generate_response(state: ChatState) -> ChatState:
    """Generate a response using the LLM."""
    print("  [generate_response] Generating response...")
    response = llm.invoke(state["messages"])
    return {
        "messages": [response],
        "current_step": "respond"
    }


def format_output(state: ChatState) -> ChatState:
    """Format the final output."""
    print("  [format_output] Formatting output...")
    return {"current_step": "complete"}


# =============================================================================
# STEP 3: Build the Graph
# =============================================================================

graph = StateGraph(ChatState)
graph.add_node("analyze", analyze_input)
graph.add_node("respond", generate_response)
graph.add_node("format", format_output)

graph.add_edge(START, "analyze")
graph.add_edge("analyze", "respond")
graph.add_edge("respond", "format")
graph.add_edge("format", END)

app = graph.compile()


# =============================================================================
# DEMO 1: Streaming with "values" mode
# =============================================================================
def demo_values_streaming():
    """Stream complete state after each node."""
    print("=" * 70)
    print("DEMO 1: Streaming with 'values' mode")
    print("=" * 70)
    print("(Shows complete state after each node)\n")

    initial_state = {
        "messages": [HumanMessage(content="Tell me a short joke about programming")],
        "current_step": "start"
    }

    for i, state in enumerate(app.stream(initial_state, stream_mode="values")):
        print(f"\n--- State Update {i + 1} ---")
        print(f"Current Step: {state.get('current_step', 'N/A')}")
        if state.get("messages"):
            last_msg = state["messages"][-1]
            content = last_msg.content if hasattr(last_msg, "content") else str(last_msg)
            print(f"Last Message: {content[:100]}{'...' if len(content) > 100 else ''}")


# =============================================================================
# DEMO 2: Streaming with "updates" mode
# =============================================================================
def demo_updates_streaming():
    """Stream only the changes made by each node."""
    print("\n" + "=" * 70)
    print("DEMO 2: Streaming with 'updates' mode")
    print("=" * 70)
    print("(Shows only changes from each node)\n")

    initial_state = {
        "messages": [HumanMessage(content="What's 2 + 2?")],
        "current_step": "start"
    }

    for node_name, updates in app.stream(initial_state, stream_mode="updates"):
        print(f"\n--- Node: {node_name} ---")
        print(f"Updates: {updates}")


# =============================================================================
# DEMO 3: Async Token Streaming
# =============================================================================
async def demo_token_streaming():
    """Stream individual tokens from LLM responses."""
    print("\n" + "=" * 70)
    print("DEMO 3: Token-by-Token Streaming")
    print("=" * 70)
    print("(Real-time token streaming)\n")

    initial_state = {
        "messages": [HumanMessage(content="Write a haiku about Python programming")],
        "current_step": "start"
    }

    print("Streaming tokens: ", end="", flush=True)

    async for event in app.astream_events(initial_state, version="v2"):
        # Filter for LLM token events
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                print(chunk.content, end="", flush=True)

    print("\n")


# =============================================================================
# DEMO 4: Custom Streaming Handler
# =============================================================================
class StreamingHandler:
    """Custom handler for processing streaming events."""

    def __init__(self):
        self.tokens = []
        self.node_updates = []

    async def process_stream(self, graph, initial_state):
        """Process streaming events with custom handling."""
        async for event in graph.astream_events(initial_state, version="v2"):
            event_type = event["event"]

            if event_type == "on_chain_start":
                # A node is starting
                if "name" in event:
                    self.node_updates.append(f"Starting: {event['name']}")

            elif event_type == "on_chat_model_stream":
                # Token from LLM
                chunk = event["data"]["chunk"]
                if hasattr(chunk, "content") and chunk.content:
                    self.tokens.append(chunk.content)
                    yield {"type": "token", "content": chunk.content}

            elif event_type == "on_chain_end":
                # A node completed
                if "name" in event:
                    self.node_updates.append(f"Completed: {event['name']}")
                    yield {"type": "node_complete", "name": event["name"]}


async def demo_custom_handler():
    """Demonstrate custom streaming handler."""
    print("\n" + "=" * 70)
    print("DEMO 4: Custom Streaming Handler")
    print("=" * 70)

    handler = StreamingHandler()
    initial_state = {
        "messages": [HumanMessage(content="Say hello in 3 languages")],
        "current_step": "start"
    }

    print("\nProcessing with custom handler:")
    async for update in handler.process_stream(app, initial_state):
        if update["type"] == "token":
            print(update["content"], end="", flush=True)
        elif update["type"] == "node_complete":
            print(f"\n  [Completed: {update['name']}]")

    print(f"\n\nTotal tokens collected: {len(handler.tokens)}")
    print(f"Node updates: {handler.node_updates}")


# =============================================================================
# DEMO 5: Streaming for Chat UI
# =============================================================================
async def stream_chat_response(user_message: str):
    """Stream a chat response - ready for UI integration."""
    initial_state = {
        "messages": [HumanMessage(content=user_message)],
        "current_step": "start"
    }

    response_text = ""

    async for event in app.astream_events(initial_state, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"]
            if hasattr(chunk, "content") and chunk.content:
                response_text += chunk.content
                # In a real UI, you would yield this to the frontend
                yield chunk.content

    return response_text


async def demo_chat_ui_streaming():
    """Simulate a chat UI with streaming."""
    print("\n" + "=" * 70)
    print("DEMO 5: Chat UI Streaming Simulation")
    print("=" * 70)

    user_input = "Explain recursion in one sentence"
    print(f"\nUser: {user_input}")
    print("Assistant: ", end="", flush=True)

    async for token in stream_chat_response(user_input):
        print(token, end="", flush=True)
        await asyncio.sleep(0.01)  # Simulate network delay

    print("\n")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # Synchronous demos
    demo_values_streaming()
    demo_updates_streaming()

    # Async demos
    print("\n" + "=" * 70)
    print("ASYNC STREAMING DEMOS")
    print("=" * 70)

    asyncio.run(demo_token_streaming())
    asyncio.run(demo_custom_handler())
    asyncio.run(demo_chat_ui_streaming())

    # Best practices
    print("=" * 70)
    print("STREAMING BEST PRACTICES")
    print("=" * 70)
    print("""
    1. Use 'values' mode when you need complete state snapshots
    2. Use 'updates' mode for efficiency with large states
    3. Use astream_events() for token-level streaming
    4. Always handle the 'version' parameter in astream_events()
    5. Filter events by type for specific use cases
    6. Consider network latency in UI implementations
    7. Buffer tokens for smoother display in chat UIs
    8. Handle interrupts and cancellation gracefully
    """)
