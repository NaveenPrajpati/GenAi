from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
import math


class EquationState(TypedDict):
    a: int
    b: int
    c: int
    equation: str
    discriminant: float
    result: str
    has_error: bool


def printEquation(state: EquationState):
    """Get coefficients from user input with validation"""
    try:
        inputA = int(input("Enter value of a: "))
        inputB = int(input("Enter value of b: "))
        inputC = int(input("Enter value of c: "))

        # Validate that 'a' is not zero (must be quadratic)
        if inputA == 0:
            return {
                "result": "Error: 'a' cannot be zero for a quadratic equation",
                "has_error": True,
            }

        equation_str = f"{inputA}x² + {inputB}x + {inputC} = 0"
        return {
            "a": inputA,
            "b": inputB,
            "c": inputC,
            "equation": equation_str,
            "has_error": False,
        }
    except ValueError:
        return {"result": "Error: Please enter valid integer values", "has_error": True}


def calculateDis(state: EquationState):
    """Calculate discriminant"""
    a = state["a"]
    b = state["b"]
    c = state["c"]

    discriminant = b**2 - 4 * a * c
    return {"discriminant": discriminant}


def calculateTwoRealRoots(state: EquationState):
    """Calculate two real roots when discriminant > 0"""
    a = state["a"]
    b = state["b"]
    discriminant = state["discriminant"]

    sqrt_d = math.sqrt(discriminant)
    r1 = (-b - sqrt_d) / (2 * a)
    r2 = (-b + sqrt_d) / (2 * a)
    return {"result": f"Two real roots: {r1:.4f} and {r2:.4f}"}


def calculateOneRealRoot(state: EquationState):
    """Calculate one real root when discriminant = 0"""
    a = state["a"]
    b = state["b"]

    r = -b / (2 * a)
    return {"result": f"One real root (repeated): {r:.4f}"}


def calculateComplexRoots(state: EquationState):
    """Calculate complex roots when discriminant < 0"""
    a = state["a"]
    b = state["b"]
    discriminant = state["discriminant"]

    real_part = -b / (2 * a)
    imaginary_part = math.sqrt(abs(discriminant)) / (2 * a)
    return {
        "result": f"Two complex roots: {real_part:.4f} + {imaginary_part:.4f}i and {real_part:.4f} - {imaginary_part:.4f}i"
    }


def handleError(state: EquationState):
    """Handle error cases - just return existing error message"""
    return {}


# Conditional routing functions
def route_after_input(state: EquationState) -> str:
    """Route based on input validation"""
    if state.get("has_error", False):
        return "error"
    return "calculate_discriminant"


def route_after_discriminant(state: EquationState) -> str:
    """Route based on discriminant value"""
    discriminant = state["discriminant"]

    if discriminant > 0:
        return "two_real_roots"
    elif discriminant == 0:
        return "one_real_root"
    else:  # discriminant < 0
        return "complex_roots"


# Create conditional workflow
graph = StateGraph(EquationState)

# Add nodes
graph.add_node("equation", printEquation)
graph.add_node("calculate_discriminant", calculateDis)
graph.add_node("two_real_roots", calculateTwoRealRoots)
graph.add_node("one_real_root", calculateOneRealRoot)
graph.add_node("complex_roots", calculateComplexRoots)
graph.add_node("error", handleError)

# Add conditional edges
graph.add_edge(START, "equation")

# First conditional: Check for input errors
graph.add_conditional_edges(
    "equation",
    route_after_input,
    {"error": "error", "calculate_discriminant": "calculate_discriminant"},
)

# Second conditional: Route based on discriminant value
graph.add_conditional_edges(
    "calculate_discriminant",
    route_after_discriminant,
    {
        "two_real_roots": "two_real_roots",
        "one_real_root": "one_real_root",
        "complex_roots": "complex_roots",
    },
)

# All terminal nodes go to END
graph.add_edge("two_real_roots", END)
graph.add_edge("one_real_root", END)
graph.add_edge("complex_roots", END)
graph.add_edge("error", END)

# Compile the graph
app = graph.compile()

# Example usage:
if __name__ == "__main__":
    # Run the graph
    result = app.invoke({})
    print(f"\nEquation: {result.get('equation', 'N/A')}")
    print(f"Result: {result['result']}")
