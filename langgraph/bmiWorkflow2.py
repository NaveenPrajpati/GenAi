from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict
from typing import Optional, Literal
import logging

# Set up logging for better debugging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BMIState(TypedDict):
    """State schema for BMI calculation workflow."""

    weight: float
    height: float
    bmi: Optional[float]
    category: Optional[str]
    result: Optional[str]
    error: Optional[str]


def validate_inputs(state: BMIState) -> BMIState:
    """Validate input parameters before calculation."""
    logger.info("Validating inputs")

    weight = state.get("weight")
    height = state.get("height")

    # Input validation
    if weight is None or height is None:
        state["error"] = "Missing weight or height"
        return state

    if weight <= 0:
        state["error"] = "Weight must be positive"
        return state

    if height <= 0:
        state["error"] = "Height must be positive"
        return state

    # Clear any previous error
    state["error"] = None
    logger.info(f"Inputs validated: weight={weight}kg, height={height}m")
    return state


def calculate_bmi(state: BMIState) -> BMIState:
    """Calculate BMI from weight and height."""
    # Skip calculation if there's an error
    if state.get("error"):
        return state

    logger.info("Calculating BMI")

    weight = state["weight"]
    height = state["height"]

    try:
        bmi = weight / (height**2)
        state["bmi"] = round(bmi, 2)
        logger.info(f"BMI calculated: {state['bmi']}")
    except Exception as e:
        state["error"] = f"Error calculating BMI: {str(e)}"
        logger.error(state["error"])

    return state


def categorize_bmi(state: BMIState) -> BMIState:
    """Categorize BMI according to WHO standards."""
    if state.get("error"):
        return state

    logger.info("Categorizing BMI")

    bmi = state.get("bmi")
    if bmi is None:
        state["error"] = "BMI not calculated"
        return state

    # WHO BMI categories
    if bmi < 18.5:
        category = "Underweight"
        result = "underweight"
    elif 18.5 <= bmi < 25.0:
        category = "Normal weight"
        result = "normal"
    elif 25.0 <= bmi < 30.0:
        category = "Overweight"
        result = "overweight"
    else:  # bmi >= 30.0
        category = "Obese"
        result = "obese"

    state["category"] = category
    state["result"] = result

    logger.info(f"BMI category: {category}")
    return state


def format_results(state: BMIState) -> BMIState:
    """Format the final results for output."""
    if state.get("error"):
        logger.error(f"Workflow completed with error: {state['error']}")
        return state

    logger.info("Formatting results")

    # Add any additional formatting or summary here
    bmi = state.get("bmi")
    category = state.get("category")

    if bmi and category:
        logger.info(f"Final result: BMI {bmi} ({category})")

    return state


def build_bmi_workflow() -> StateGraph:
    """Build and compile the BMI calculation workflow."""
    logger.info("Building BMI workflow")

    # Create the state graph
    graph = StateGraph(BMIState)

    # Add nodes
    graph.add_node("validate_inputs", validate_inputs)
    graph.add_node("calculate_bmi", calculate_bmi)
    graph.add_node("categorize_bmi", categorize_bmi)
    graph.add_node("format_results", format_results)

    # Add edges to define the workflow
    graph.add_edge(START, "validate_inputs")
    graph.add_edge("validate_inputs", "calculate_bmi")
    graph.add_edge("calculate_bmi", "categorize_bmi")
    graph.add_edge("categorize_bmi", "format_results")
    graph.add_edge("format_results", END)

    # Compile and return the workflow
    workflow = graph.compile()
    logger.info("BMI workflow compiled successfully")

    return workflow


def main():
    """Main function to demonstrate the BMI calculator."""
    workflow = build_bmi_workflow()

    inputWeight = float(input("Enter Weight :- "))
    inputHeight = float(input("Enter Height :- "))

    print("=== BMI Calculator Test Results ===\n")

    try:
        result = workflow.invoke({"weight": inputWeight, "height": inputHeight})
        if result.get("error"):
            print(f"Error: {result['error']}")
        else:
            bmi = result.get("bmi")
            category = result.get("category")
            print(f"BMI: {bmi} ({category})")
    except Exception as e:
        print(f"Workflow error: {str(e)}")

    # Visualize the graph (if running in Jupyter)
    try:
        from IPython.display import Image, display

        print("\nWorkflow Visualization:")
        display(Image(workflow.get_graph().draw_mermaid_png()))
    except ImportError:
        print("\nTo visualize the graph, install IPython and run in Jupyter:")
        print("pip install ipython")
        print("# Then in Jupyter:")
        print("from IPython.display import Image, display")
        print("display(Image(workflow.get_graph().draw_mermaid_png()))")


if __name__ == "__main__":
    main()
