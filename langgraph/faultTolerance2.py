"""
Enhanced LangGraph Fault Tolerance Example
==========================================

This example demonstrates how LangGraph handles interruptions and resumes execution
from checkpoints, showcasing fault tolerance capabilities.

Key Concepts Demonstrated:
1. State persistence with checkpointing
2. Graceful handling of interruptions
3. Resuming execution from the last successful checkpoint
4. State history tracking
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Optional
import time
import signal
import sys
from datetime import datetime


class ProcessingState(TypedDict):
    """
    State schema for our processing pipeline.

    Fields:
        input: Initial input data
        step1_result: Result from step 1 processing
        step2_result: Result from step 2 processing
        step3_result: Result from step 3 processing
        processing_start_time: Timestamp when processing began
        current_step: Track which step we're currently on
    """

    input: str
    step1_result: Optional[str]
    step2_result: Optional[str]
    step3_result: Optional[str]
    processing_start_time: Optional[str]
    current_step: Optional[str]


def log_step_execution(step_name: str, message: str):
    """Helper function to log step execution with timestamps."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {step_name}: {message}")


def step_1_data_preparation(state: ProcessingState) -> ProcessingState:
    """
    Step 1: Data preparation and validation.
    This step simulates preprocessing input data.
    """
    log_step_execution("STEP 1", "Starting data preparation...")

    # Simulate some processing time
    time.sleep(1)

    processed_input = f"processed_{state['input']}"
    log_step_execution("STEP 1", f"Data prepared: {processed_input}")

    return {
        "step1_result": processed_input,
        "current_step": "step_1_completed",
        "processing_start_time": state.get("processing_start_time")
        or datetime.now().isoformat(),
    }


def step_2_heavy_computation(state: ProcessingState) -> ProcessingState:
    """
    Step 2: Heavy computation that might fail or be interrupted.
    This step simulates a long-running process that could crash.
    """
    log_step_execution("STEP 2", "Starting heavy computation...")
    log_step_execution(
        "STEP 2", "This step will hang - interrupt it manually to test fault tolerance"
    )
    log_step_execution(
        "STEP 2", "In Jupyter: use 'Interrupt' button | In terminal: use Ctrl+C"
    )

    try:
        # Simulate long-running computation that might be interrupted
        for i in range(100):
            time.sleep(0.5)  # Shorter sleep intervals for more responsive interruption
            if i % 10 == 0:
                log_step_execution("STEP 2", f"Processing... {i}% complete")

        # If we reach here, the computation completed successfully
        result = f"computation_result_for_{state['step1_result']}"
        log_step_execution("STEP 2", f"Heavy computation completed: {result}")

        return {"step2_result": result, "current_step": "step_2_completed"}

    except Exception as e:
        log_step_execution("STEP 2", f"Error during computation: {e}")
        raise


def step_3_finalization(state: ProcessingState) -> ProcessingState:
    """
    Step 3: Finalize results and cleanup.
    This step only runs if previous steps completed successfully.
    """
    log_step_execution("STEP 3", "Starting finalization...")

    # Simulate finalization processing
    time.sleep(1)

    final_result = f"final_{state['step2_result']}_complete"
    log_step_execution("STEP 3", f"Processing pipeline completed: {final_result}")

    return {"step3_result": final_result, "current_step": "pipeline_completed"}


def create_fault_tolerant_graph():
    """
    Creates and configures the fault-tolerant processing graph.

    Returns:
        Compiled graph with checkpointing enabled
    """
    # Create the state graph
    builder = StateGraph(ProcessingState)

    # Add processing nodes
    builder.add_node("data_prep", step_1_data_preparation)
    builder.add_node("heavy_compute", step_2_heavy_computation)
    builder.add_node("finalize", step_3_finalization)

    # Define the processing flow
    builder.set_entry_point("data_prep")
    builder.add_edge("data_prep", "heavy_compute")
    builder.add_edge("heavy_compute", "finalize")
    builder.add_edge("finalize", END)

    # Enable checkpointing for fault tolerance
    checkpointer = InMemorySaver()
    graph = builder.compile(checkpointer=checkpointer)

    return graph


def demonstrate_fault_tolerance():
    """
    Main demonstration function showing fault tolerance capabilities.
    """
    print("=" * 60)
    print("LangGraph Fault Tolerance Demonstration")
    print("=" * 60)

    graph = create_fault_tolerant_graph()
    thread_id = "fault-tolerance-demo"
    config = {"configurable": {"thread_id": thread_id}}

    # Initial execution attempt
    print("\n🚀 PHASE 1: Initial execution attempt")
    print("-" * 40)

    try:
        initial_state = {
            "input": "sample_data",
            "step1_result": None,
            "step2_result": None,
            "step3_result": None,
            "processing_start_time": None,
            "current_step": "starting",
        }

        result = graph.invoke(initial_state, config=config)
        print(f"\n✅ Pipeline completed successfully: {result}")

    except KeyboardInterrupt:
        print(f"\n❌ Pipeline interrupted by user")
        print("💾 State has been automatically saved to checkpoint")

        # Show current state after interruption
        current_state = graph.get_state(config)
        print(f"📊 Current state: {current_state.values}")
        print(f"🔄 Next step to execute: {current_state.next}")

    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        return

    # Recovery demonstration
    print(f"\n🔄 PHASE 2: Recovery and resumption")
    print("-" * 40)
    print("Attempting to resume from last checkpoint...")

    try:
        # Resume execution - LangGraph will continue from where it left off
        resumed_result = graph.invoke(
            None, config=config
        )  # None means "continue from checkpoint"
        print(f"\n✅ Pipeline resumed and completed: {resumed_result}")

    except Exception as e:
        print(f"\n❌ Recovery failed: {e}")
        return

    # Show execution history
    print(f"\n📚 PHASE 3: Execution history analysis")
    print("-" * 40)

    try:
        history = list(graph.get_state_history(config))
        print(f"Total checkpoints saved: {len(history)}")

        for i, checkpoint in enumerate(history):
            step_info = checkpoint.values.get("current_step", "unknown")
            print(f"  Checkpoint {i+1}: {step_info}")

    except Exception as e:
        print(f"❌ Could not retrieve history: {e}")


def setup_signal_handler():
    """Setup graceful signal handling for demonstration purposes."""

    def signal_handler(signum, frame):
        print(f"\n🛑 Received signal {signum}")
        print(
            "📝 Note: In a real application, you might want to save additional state here"
        )
        # sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":
    # Setup signal handling
    setup_signal_handler()

    # Run the demonstration
    demonstrate_fault_tolerance()

    print(f"\n" + "=" * 60)
    print("Demonstration completed!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("• LangGraph automatically saves state at each node completion")
    print("• Interrupted executions can be resumed from the last checkpoint")
    print("• State history provides audit trail of execution")
    print("• Fault tolerance works without additional code in the node functions")
