"""
Enhanced LangGraph Persistence Example
=====================================

This example demonstrates LangGraph's persistence features including:
- State checkpointing with InMemorySaver
- Thread-based conversation management
- State history tracking
- Time travel to previous checkpoints
- State modification and continuation
"""

from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
import json

# Load environment variables for API keys
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(model="gpt-3.5-turbo")


class JokeState(TypedDict):
    """
    State schema for the joke generation workflow.

    Attributes:
        topic: The topic for joke generation
        joke: The generated joke
        explanation: Explanation of why the joke is funny
    """

    topic: str
    joke: str
    explanation: str


def generate_joke_node(state: JokeState) -> dict:
    """
    Generate a joke based on the given topic.

    Args:
        state: Current state containing the topic

    Returns:
        Dictionary with the generated joke
    """
    topic = state["topic"]
    prompt = f"Generate a clean, family-friendly joke about {topic}. Make it clever and witty."

    print(f"🎭 Generating joke for topic: {topic}")
    response = llm.invoke(prompt).content
    print(f"📝 Generated joke: {response[:50]}...")

    return {"joke": response}


def explain_joke_node(state: JokeState) -> dict:
    """
    Generate an explanation for the joke.

    Args:
        state: Current state containing the joke

    Returns:
        Dictionary with the joke explanation
    """
    joke = state["joke"]
    prompt = f"""Explain why this joke is funny in 2-3 sentences. 
    Focus on the wordplay, timing, or comedic elements:
    
    Joke: {joke}"""

    print(f"🧠 Explaining the joke...")
    response = llm.invoke(prompt).content
    print(f"📚 Generated explanation: {response[:50]}...")

    return {"explanation": response}


def create_joke_workflow():
    """Create and compile the joke generation workflow."""

    # Create the state graph
    graph = StateGraph(JokeState)

    # Add nodes to the graph
    graph.add_node("generate_joke", generate_joke_node)
    graph.add_node("explain_joke", explain_joke_node)

    # Define the workflow edges
    graph.add_edge(START, "generate_joke")
    graph.add_edge("generate_joke", "explain_joke")
    graph.add_edge("explain_joke", END)

    # Set up persistence with InMemorySaver
    checkpointer = InMemorySaver()

    # Compile the workflow with checkpointing enabled
    workflow = graph.compile(checkpointer=checkpointer)

    return workflow


def print_state_info(workflow, config, title="Current State"):
    """Helper function to print state information clearly."""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")

    current_state = workflow.get_state(config)
    print(f"Thread ID: {config['configurable']['thread_id']}")
    print(
        f"Checkpoint ID: {current_state.config['configurable'].get('checkpoint_id', 'N/A')}"
    )
    print(f"Next Step: {current_state.next}")
    print(f"State Values:")
    for key, value in current_state.values.items():
        if value:
            display_value = value[:100] + "..." if len(str(value)) > 100 else value
            print(f"  {key}: {display_value}")
    print()


def demonstrate_basic_persistence():
    """Demonstrate basic persistence functionality."""
    print("🚀 DEMONSTRATION 1: Basic Persistence")
    print("=" * 60)

    workflow = create_joke_workflow()

    # Create configuration for thread 1
    thread_1_config = {"configurable": {"thread_id": "conversation_1"}}

    print("Starting workflow for Thread 1 with topic 'pizza'...")
    result_1 = workflow.invoke({"topic": "pizza"}, config=thread_1_config)

    print_state_info(workflow, thread_1_config, "Thread 1 Final State")

    # Create configuration for thread 2
    thread_2_config = {"configurable": {"thread_id": "conversation_2"}}

    print("Starting workflow for Thread 2 with topic 'cats'...")
    result_2 = workflow.invoke({"topic": "cats"}, config=thread_2_config)

    print_state_info(workflow, thread_2_config, "Thread 2 Final State")

    # Show that states are independent
    print("Verifying Thread 1 state is still intact:")
    print_state_info(workflow, thread_1_config, "Thread 1 State (Unchanged)")

    return workflow, thread_1_config, thread_2_config


def demonstrate_state_history(workflow, config):
    """Demonstrate state history functionality."""
    print("\n🕒 DEMONSTRATION 2: State History")
    print("=" * 60)

    print("Retrieving state history for Thread 1...")
    history = list(workflow.get_state_history(config))

    print(f"Found {len(history)} checkpoints in history:")
    for i, checkpoint in enumerate(history):
        print(f"\nCheckpoint {i + 1}:")
        print(f"  ID: {checkpoint.config['configurable']['checkpoint_id']}")
        print(f"  Next: {checkpoint.next}")
        print(f"  Values: {list(checkpoint.values.keys())}")

        # Show what values are present at this checkpoint
        for key, value in checkpoint.values.items():
            if value:
                display_value = value[:50] + "..." if len(str(value)) > 50 else value
                print(f"    {key}: {display_value}")


def demonstrate_time_travel(workflow, config):
    """Demonstrate time travel to previous checkpoints."""
    print("\n⏰ DEMONSTRATION 3: Time Travel")
    print("=" * 60)

    # Get the state history to find a checkpoint to travel to
    history = list(workflow.get_state_history(config))

    if len(history) < 2:
        print("Not enough history for time travel demonstration")
        return

    # Travel to the second checkpoint (after joke generation, before explanation)
    target_checkpoint = history[1]  # Second most recent
    target_checkpoint_id = target_checkpoint.config["configurable"]["checkpoint_id"]

    print(f"Time traveling to checkpoint: {target_checkpoint_id}")

    # Create config for time travel
    time_travel_config = {
        "configurable": {
            "thread_id": config["configurable"]["thread_id"],
            "checkpoint_id": target_checkpoint_id,
        }
    }

    # Get state at that checkpoint
    past_state = workflow.get_state(time_travel_config)
    print_state_info(workflow, time_travel_config, "State at Past Checkpoint")

    # Continue execution from that point
    print("Continuing execution from the past checkpoint...")
    continued_result = workflow.invoke(None, config=time_travel_config)

    print("Execution completed. Checking current state:")
    print_state_info(workflow, config, "Current State After Time Travel")


def demonstrate_state_modification(workflow, config):
    """Demonstrate state modification and continuation."""
    print("\n✏️  DEMONSTRATION 4: State Modification")
    print("=" * 60)

    # Get current state history
    history = list(workflow.get_state_history(config))

    if len(history) < 2:
        print("Not enough history for state modification demonstration")
        return

    # Find a checkpoint where we can modify the topic
    target_checkpoint = history[1]  # After joke generation
    target_checkpoint_id = target_checkpoint.config["configurable"]["checkpoint_id"]

    print(f"Modifying state at checkpoint: {target_checkpoint_id}")
    print("Original topic was 'pizza', changing to 'robots'...")

    # Update the state
    modification_config = {
        "configurable": {
            "thread_id": config["configurable"]["thread_id"],
            "checkpoint_id": target_checkpoint_id,
            "checkpoint_ns": "",  # Empty namespace for main thread
        }
    }

    workflow.update_state(modification_config, {"topic": "robots"})

    print("State updated! New history:")
    updated_history = list(workflow.get_state_history(config))

    print(f"History now contains {len(updated_history)} checkpoints")

    # Get the new checkpoint created by the update
    if len(updated_history) > len(history):
        new_checkpoint = updated_history[0]  # Most recent
        new_checkpoint_id = new_checkpoint.config["configurable"]["checkpoint_id"]

        print(f"New checkpoint created: {new_checkpoint_id}")

        # Continue from the modified state
        continue_config = {
            "configurable": {
                "thread_id": config["configurable"]["thread_id"],
                "checkpoint_id": new_checkpoint_id,
            }
        }

        print("Continuing execution with modified state...")
        workflow.invoke(None, config=continue_config)

        print_state_info(workflow, config, "Final State After Modification")


def main():
    """Run all demonstrations."""
    print("🎪 LangGraph Persistence Features Demonstration")
    print("=" * 80)

    try:
        # Demonstration 1: Basic persistence
        workflow, thread_1_config, thread_2_config = demonstrate_basic_persistence()

        # Demonstration 2: State history
        demonstrate_state_history(workflow, thread_1_config)

        # Demonstration 3: Time travel
        demonstrate_time_travel(workflow, thread_1_config)

        # Demonstration 4: State modification
        demonstrate_state_modification(workflow, thread_1_config)

        print("\n🎉 All demonstrations completed successfully!")
        print("=" * 80)

    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        print("Make sure you have your OpenAI API key set up properly.")


if __name__ == "__main__":
    main()
