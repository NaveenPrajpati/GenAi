import os
from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
import json
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import ToolMessage
from langgraph.prebuilt import ToolNode
from dotenv import load_dotenv

load_dotenv()
# Set your API key
os.environ["ANTHROPIC_API_KEY"] = "your-api-key-here"


# Define the state schema
class State(TypedDict):
    """The state of our chatbot."""

    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (appends messages to the list rather than overwriting)
    messages: Annotated[list, add_messages]


# Initialize the chat model
llm = ChatOpenAI()


def chatbot(state: State):
    """Basic chatbot function that generates responses."""
    return {"messages": [llm.invoke(state["messages"])]}


# Create the graph
graph_builder = StateGraph(State)
graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

# Compile the graph
basic_graph = graph_builder.compile()


# Test the basic chatbot
def run_basic_chatbot():
    """Run a simple conversation loop."""
    print("Basic Chatbot (type 'quit' to exit)")
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        for event in basic_graph.stream(
            {"messages": [{"role": "user", "content": user_input}]}
        ):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)


@tool
def search_web(query: str) -> str:
    """Search the web for current information."""
    # Simple web search implementation (replace with actual API)
    # For demo purposes, we'll simulate a search result
    return f"Search results for '{query}': [Simulated search results would appear here]"


@tool
def calculate(expression: str) -> str:
    """Calculate mathematical expressions."""
    try:
        # Safe evaluation of mathematical expressions
        result = eval(expression.replace("^", "**"))
        return str(result)
    except Exception as e:
        return f"Error calculating {expression}: {str(e)}"


@tool
def get_weather(location: str) -> str:
    """Get current weather for a location."""
    # Simulated weather API call
    return f"Current weather in {location}: Partly cloudy, 72°F"


# Define tools
tools = [search_web, calculate, get_weather]
tool_node = ToolNode(tools)


# Enhanced state with tools
class ToolState(TypedDict):
    messages: Annotated[list, add_messages]


def should_continue(state: ToolState):
    """Determine if we should continue or end."""
    messages = state["messages"]
    last_message = messages[-1]
    # If the LLM makes a tool call, then we route to the "tools" node
    if last_message.tool_calls:
        return "tools"
    # Otherwise, we stop (reply to the user)
    return END


def call_model(state: ToolState):
    """Call the LLM with tool capabilities."""
    messages = state["messages"]
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


# Create enhanced graph with tools
tool_graph_builder = StateGraph(ToolState)
tool_graph_builder.add_node("agent", call_model)
tool_graph_builder.add_node("tools", tool_node)

tool_graph_builder.add_edge(START, "agent")
tool_graph_builder.add_conditional_edges("agent", should_continue)
tool_graph_builder.add_edge("tools", "agent")

tool_graph = tool_graph_builder.compile()


def run_tool_chatbot():
    """Run chatbot with tools."""
    print("Tool-Enhanced Chatbot (type 'quit' to exit)")
    print(
        "Try: 'Calculate 15 * 7', 'Search for Python tutorials', 'What's the weather in Paris?'"
    )

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        for event in tool_graph.stream(
            {"messages": [{"role": "user", "content": user_input}]}
        ):
            for value in event.values():
                if "messages" in value:
                    for message in value["messages"]:
                        if hasattr(message, "content") and message.content:
                            print("Assistant:", message.content)


from langgraph.checkpoint.sqlite import SqliteSaver

# Create memory/checkpointer
memory = SqliteSaver.from_conn_string(":memory:")


class MemoryState(TypedDict):
    messages: Annotated[list, add_messages]
    conversation_id: str


def call_model_with_memory(state: MemoryState):
    """Call the LLM with persistent memory."""
    messages = state["messages"]
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue_memory(state: MemoryState):
    """Determine if we should continue or end with memory."""
    messages = state["messages"]
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END


# Create graph with memory
memory_graph_builder = StateGraph(MemoryState)
memory_graph_builder.add_node("agent", call_model_with_memory)
memory_graph_builder.add_node("tools", tool_node)

memory_graph_builder.add_edge(START, "agent")
memory_graph_builder.add_conditional_edges("agent", should_continue_memory)
memory_graph_builder.add_edge("tools", "agent")

# Compile with checkpointer for memory
memory_graph = memory_graph_builder.compile(checkpointer=memory)


def run_memory_chatbot():
    """Run chatbot with persistent memory."""
    print("Memory-Enabled Chatbot (type 'quit' to exit)")
    print("This bot remembers our conversation!")

    config = {"configurable": {"thread_id": "conversation-1"}}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        for event in memory_graph.stream(
            {"messages": [{"role": "user", "content": user_input}]}, config
        ):
            for value in event.values():
                if "messages" in value:
                    for message in value["messages"]:
                        if hasattr(message, "content") and message.content:
                            print("Assistant:", message.content)


from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage


class HITLState(TypedDict):
    messages: Annotated[list, add_messages]
    pending_action: str
    approved: bool


def requires_approval(action: str) -> bool:
    """Check if an action requires human approval."""
    sensitive_actions = ["search_web", "calculate", "get_weather"]
    return any(action.startswith(sa) for sa in sensitive_actions)


def call_model_hitl(state: HITLState):
    """Call model with human-in-the-loop checks."""
    messages = state["messages"]
    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(messages)

    # Check if response contains tool calls that need approval
    if response.tool_calls:
        tool_call = response.tool_calls[0]
        if requires_approval(tool_call["name"]):
            return {
                "messages": [response],
                "pending_action": f"Tool: {tool_call['name']}, Args: {tool_call['args']}",
                "approved": False,
            }

    return {"messages": [response], "pending_action": "", "approved": True}


def human_approval(state: HITLState):
    """Request human approval for sensitive actions."""
    if state.get("pending_action"):
        print(f"\n🤔 The assistant wants to: {state['pending_action']}")
        approval = input("Approve this action? (y/n): ").lower().strip()

        if approval in ["y", "yes"]:
            return {"approved": True}
        else:
            return {
                "approved": False,
                "messages": [HumanMessage(content="Action was not approved by human.")],
            }
    return {"approved": True}


def execute_tools_hitl(state: HITLState):
    """Execute tools only if approved."""
    if state.get("approved", False):
        messages = state["messages"]
        last_message = messages[-1]
        if last_message.tool_calls:
            # Execute the tool
            tool_call = last_message.tool_calls[0]
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            # Find and execute the tool
            for tool in tools:
                if tool.name == tool_name:
                    result = tool.invoke(tool_args)
                    return {
                        "messages": [
                            ToolMessage(
                                content=str(result), tool_call_id=tool_call["id"]
                            )
                        ]
                    }

    return {"messages": []}


def should_continue_hitl(state: HITLState):
    """Routing logic for HITL flow."""
    messages = state["messages"]
    if not messages:
        return END

    last_message = messages[-1]

    if last_message.tool_calls and not state.get("approved", False):
        return "human_approval"
    elif last_message.tool_calls and state.get("approved", False):
        return "execute_tools"
    else:
        return END


# Create HITL graph
hitl_graph_builder = StateGraph(HITLState)
hitl_graph_builder.add_node("agent", call_model_hitl)
hitl_graph_builder.add_node("human_approval", human_approval)
hitl_graph_builder.add_node("execute_tools", execute_tools_hitl)

hitl_graph_builder.add_edge(START, "agent")
hitl_graph_builder.add_conditional_edges("agent", should_continue_hitl)
hitl_graph_builder.add_edge("human_approval", "execute_tools")
hitl_graph_builder.add_edge("execute_tools", "agent")

hitl_graph = hitl_graph_builder.compile(checkpointer=memory)


def run_hitl_chatbot():
    """Run chatbot with human-in-the-loop approval."""
    print("Human-in-the-Loop Chatbot (type 'quit' to exit)")
    print("This bot will ask for approval before taking actions!")

    config = {"configurable": {"thread_id": "hitl-conversation-1"}}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        for event in hitl_graph.stream(
            {"messages": [{"role": "user", "content": user_input}]}, config
        ):
            for value in event.values():
                if "messages" in value and value["messages"]:
                    for message in value["messages"]:
                        if (
                            hasattr(message, "content")
                            and message.content
                            and not message.content.startswith("Action was not")
                        ):
                            print("Assistant:", message.content)


from typing import Dict, List
import operator


class CustomState(TypedDict):
    messages: Annotated[list, add_messages]
    user_preferences: Dict[str, str]
    conversation_summary: str
    tool_usage_count: Annotated[int, operator.add]
    context: Dict[str, str]


def update_preferences(state: CustomState, user_input: str):
    """Extract and update user preferences from conversation."""
    preferences = state.get("user_preferences", {})

    # Simple preference extraction (in practice, you'd use more sophisticated NLP)
    if "i like" in user_input.lower():
        # Extract preference
        preference = user_input.lower().split("i like")[1].strip()
        preferences["likes"] = preference
    elif "i prefer" in user_input.lower():
        preference = user_input.lower().split("i prefer")[1].strip()
        preferences["prefers"] = preference

    return preferences


def call_model_custom(state: CustomState):
    """Enhanced model call with custom state management."""
    messages = state["messages"]
    preferences = state.get("user_preferences", {})
    context = state.get("context", {})

    # Add preferences to system message if they exist
    enhanced_messages = list(messages)
    if preferences:
        system_msg = (
            f"User preferences: {preferences}. Keep these in mind when responding."
        )
        enhanced_messages.insert(0, {"role": "system", "content": system_msg})

    model_with_tools = llm.bind_tools(tools)
    response = model_with_tools.invoke(enhanced_messages)

    # Update tool usage count
    tool_count_increment = 1 if response.tool_calls else 0

    # Extract user preferences from the latest message
    user_msg = messages[-1].content if messages else ""
    updated_preferences = update_preferences(state, user_msg)

    return {
        "messages": [response],
        "user_preferences": updated_preferences,
        "tool_usage_count": tool_count_increment,
        "context": {
            "last_response_type": "tool_call" if response.tool_calls else "text"
        },
    }


def should_continue_custom(state: CustomState):
    """Enhanced routing with state awareness."""
    messages = state["messages"]
    last_message = messages[-1]
    tool_count = state.get("tool_usage_count", 0)

    if last_message.tool_calls and tool_count < 5:  # Limit tool usage
        return "tools"
    elif tool_count >= 5:
        return "summarize"
    return END


def summarize_conversation(state: CustomState):
    """Create a conversation summary."""
    messages = state["messages"]
    summary_prompt = f"Summarize this conversation in 2-3 sentences: {messages[-5:]}"
    summary = llm.invoke([{"role": "user", "content": summary_prompt}])

    return {
        "conversation_summary": summary.content,
        "tool_usage_count": 0,  # Reset counter after summary
    }


# Create custom state graph
custom_graph_builder = StateGraph(CustomState)
custom_graph_builder.add_node("agent", call_model_custom)
custom_graph_builder.add_node("tools", tool_node)
custom_graph_builder.add_node("summarize", summarize_conversation)

custom_graph_builder.add_edge(START, "agent")
custom_graph_builder.add_conditional_edges("agent", should_continue_custom)
custom_graph_builder.add_edge("tools", "agent")
custom_graph_builder.add_edge("summarize", END)

custom_graph = custom_graph_builder.compile(checkpointer=memory)


def run_custom_chatbot():
    """Run chatbot with custom state management."""
    print("Custom State Chatbot (type 'quit' to exit)")
    print("This bot learns your preferences and tracks conversation context!")

    config = {"configurable": {"thread_id": "custom-conversation-1"}}

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        for event in custom_graph.stream(
            {
                "messages": [{"role": "user", "content": user_input}],
                "user_preferences": {},
                "tool_usage_count": 0,
                "context": {},
            },
            config,
        ):
            for value in event.values():
                if "messages" in value and value["messages"]:
                    for message in value["messages"]:
                        if hasattr(message, "content") and message.content:
                            print("Assistant:", message.content)

                # Display state information
                if "user_preferences" in value and value["user_preferences"]:
                    print(f"📝 Updated preferences: {value['user_preferences']}")
                if "conversation_summary" in value and value["conversation_summary"]:
                    print(f"📋 Conversation summary: {value['conversation_summary']}")


import datetime
from typing import Optional


class TimeTravel:
    """Time travel functionality for LangGraph conversations."""

    def __init__(self, graph, checkpointer):
        self.graph = graph
        self.checkpointer = checkpointer

    def get_conversation_history(self, thread_id: str) -> List[Dict]:
        """Get all checkpoints for a conversation thread."""
        config = {"configurable": {"thread_id": thread_id}}
        history = []

        for state in self.graph.get_state_history(config):
            history.append(
                {
                    "checkpoint_id": state.config["configurable"]["checkpoint_id"],
                    "timestamp": state.created_at,
                    "messages": state.values.get("messages", []),
                    "metadata": state.metadata,
                }
            )

        return history

    def rewind_to_checkpoint(self, thread_id: str, checkpoint_id: str):
        """Rewind conversation to a specific checkpoint."""
        config = {
            "configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}
        }
        return self.graph.get_state(config)

    def branch_conversation(
        self, thread_id: str, checkpoint_id: str, new_thread_id: str
    ):
        """Create a new conversation branch from a checkpoint."""
        # Get state at checkpoint
        old_config = {
            "configurable": {"thread_id": thread_id, "checkpoint_id": checkpoint_id}
        }
        state = self.graph.get_state(old_config)

        # Create new thread with this state
        new_config = {"configurable": {"thread_id": new_thread_id}}
        return self.graph.update_state(new_config, state.values)


def run_time_travel_chatbot():
    """Run chatbot with time travel capabilities."""
    print("Time Travel Chatbot (type 'quit' to exit)")
    print("Special commands:")
    print("  'history' - Show conversation checkpoints")
    print("  'rewind <checkpoint_id>' - Go back to a checkpoint")
    print("  'branch <checkpoint_id> <new_thread>' - Create conversation branch")

    # Use the custom graph with time travel
    time_travel = TimeTravel(custom_graph, memory)
    current_thread = "time-travel-conversation"

    config = {"configurable": {"thread_id": current_thread}}

    while True:
        user_input = input("User: ")

        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break
        elif user_input.lower() == "history":
            history = time_travel.get_conversation_history(current_thread)
            print("\n🕐 Conversation History:")
            for i, checkpoint in enumerate(history[-5:]):  # Show last 5
                timestamp = checkpoint.get("timestamp", "Unknown")
                msg_count = len(checkpoint.get("messages", []))
                print(f"  {i}: {timestamp} - {msg_count} messages")
            continue
        elif user_input.lower().startswith("rewind"):
            try:
                _, checkpoint_id = user_input.split(maxsplit=1)
                state = time_travel.rewind_to_checkpoint(current_thread, checkpoint_id)
                print(f"🔄 Rewound to checkpoint: {checkpoint_id}")
                print(
                    f"Messages at that point: {len(state.values.get('messages', []))}"
                )
            except Exception as e:
                print(f"❌ Error rewinding: {e}")
            continue
        elif user_input.lower().startswith("branch"):
            try:
                _, checkpoint_id, new_thread = user_input.split(maxsplit=2)
                time_travel.branch_conversation(
                    current_thread, checkpoint_id, new_thread
                )
                current_thread = new_thread
                config = {"configurable": {"thread_id": current_thread}}
                print(f"🌿 Created new branch: {new_thread}")
            except Exception as e:
                print(f"❌ Error creating branch: {e}")
            continue

        # Normal conversation
        for event in custom_graph.stream(
            {
                "messages": [{"role": "user", "content": user_input}],
                "user_preferences": {},
                "tool_usage_count": 0,
                "context": {},
            },
            config,
        ):
            for value in event.values():
                if "messages" in value and value["messages"]:
                    for message in value["messages"]:
                        if hasattr(message, "content") and message.content:
                            print("Assistant:", message.content)


## Running the Complete System

# Choose which chatbot to run:
if __name__ == "__main__":
    print("Choose a chatbot to run:")
    print("1. Basic Chatbot")
    print("2. Tool-Enhanced Chatbot")
    print("3. Memory-Enabled Chatbot")
    print("4. Human-in-the-Loop Chatbot")
    print("5. Custom State Chatbot")
    print("6. Time Travel Chatbot")

    choice = input("Enter choice (1-6): ")

    if choice == "1":
        run_basic_chatbot()
    elif choice == "2":
        run_tool_chatbot()
    elif choice == "3":
        run_memory_chatbot()
    elif choice == "4":
        run_hitl_chatbot()
    elif choice == "5":
        run_custom_chatbot()
    elif choice == "6":
        run_time_travel_chatbot()
    else:
        print("Invalid choice. Running basic chatbot...")
        run_basic_chatbot()


# Uncomment to run
# run_basic_chatbot()
