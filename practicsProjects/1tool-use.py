"""
Tool Use Agent with Guardrails
===============================

LEARNING OBJECTIVES:
- Build a production-ready tool-using agent
- Implement input guardrails for safety
- Use safe mathematical evaluation (no eval!)
- Handle tool routing and error cases

CONCEPT:
This agent demonstrates:
1. Custom tool creation with @tool decorator
2. Input validation and guardrails
3. Safe expression evaluation using AST
4. Conditional routing based on input analysis

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Tool Agent Flow                           │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │   User Input                                                │
    │        │                                                    │
    │        ▼                                                    │
    │   ┌──────────────┐                                         │
    │   │  Guardrails  │ ──► Block dangerous requests            │
    │   └──────┬───────┘                                         │
    │          │                                                  │
    │          ▼                                                  │
    │   ┌──────────────┐                                         │
    │   │ Analyze Tool │ ──► Detect which tool is needed         │
    │   └──────┬───────┘                                         │
    │          │                                                  │
    │    ┌─────┴─────┬───────────────┐                           │
    │    ▼           ▼               ▼                            │
    │ Calculator  Dictionary    DateTime                          │
    │    │           │               │                            │
    │    └─────┬─────┴───────────────┘                           │
    │          ▼                                                  │
    │      Response                                               │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

PREREQUISITES:
- Completed: tools/customTools.py, tools/toolbinding.py
- Understanding of LangGraph StateGraph
- OpenAI API key in .env

NEXT STEPS:
- practicsProjects/2ragPdf.py - RAG pipeline
- practicsProjects/6multiAgent.py - Multi-agent systems
"""

from __future__ import annotations
from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import AIMessage, HumanMessage, BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
import ast
import operator
import math
import re
from IPython.display import Image, display
from dotenv import load_dotenv


load_dotenv()


# =============================================================================
# SAFE MATH EVALUATOR (No eval()!)
# =============================================================================
class SafeMathEvaluator:
    """
    Safely evaluate mathematical expressions using AST parsing.
    This is MUCH safer than eval() as it only allows specific operations.
    """

    # Allowed operators
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
    }

    # Allowed functions
    FUNCTIONS = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'abs': abs,
        'round': round,
        'floor': math.floor,
        'ceil': math.ceil,
    }

    # Allowed constants
    CONSTANTS = {
        'pi': math.pi,
        'e': math.e,
    }

    def evaluate(self, expression: str) -> float:
        """Safely evaluate a mathematical expression."""
        try:
            tree = ast.parse(expression, mode='eval')
            return self._eval_node(tree.body)
        except Exception as e:
            raise ValueError(f"Invalid expression: {e}")

    def _eval_node(self, node):
        """Recursively evaluate AST nodes."""
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)):
                return node.value
            raise ValueError(f"Invalid constant: {node.value}")

        elif isinstance(node, ast.Name):
            if node.id in self.CONSTANTS:
                return self.CONSTANTS[node.id]
            raise ValueError(f"Unknown variable: {node.id}")

        elif isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type}")
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            return self.OPERATORS[op_type](left, right)

        elif isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in self.OPERATORS:
                raise ValueError(f"Unsupported operator: {op_type}")
            operand = self._eval_node(node.operand)
            return self.OPERATORS[op_type](operand)

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                func_name = node.func.id
                if func_name in self.FUNCTIONS:
                    args = [self._eval_node(arg) for arg in node.args]
                    return self.FUNCTIONS[func_name](*args)
            raise ValueError(f"Unknown function: {node.func}")

        else:
            raise ValueError(f"Unsupported node type: {type(node)}")


# Create global evaluator instance
safe_eval = SafeMathEvaluator()


# =============================================================================
# TOOLS
# =============================================================================
@tool
def calculator(expression: str) -> str:
    """Evaluate a simple arithmetic expression like '17% of 350' or '2+2'.

    Supports: +, -, *, /, **, %, sqrt, sin, cos, tan, log, pi, e
    """
    try:
        # Handle percentage operations specially
        if "of" in expression and "%" in expression:
            match = re.match(
                r"(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:\.\d+)?)", expression.strip()
            )
            if match:
                percentage = float(match.group(1))
                value = float(match.group(2))
                result = (percentage / 100) * value
                return f"{expression} = {result}"

        # Use safe evaluator for all other expressions
        result = safe_eval.evaluate(expression)
        return f"{expression} = {result}"

    except Exception as e:
        return f"Error calculating '{expression}': {str(e)}"


@tool
def dictionary_lookup(word: str) -> str:
    """Return the definition of a term from a local mini-dictionary."""
    data = {
        "serendipity": "the occurrence of events by chance in a beneficial way",
        "idempotent": "an operation that can be applied multiple times without changing the result",
    }
    word = word.lower().strip()
    if word in data:
        return f"{word}: {data[word]}"
    else:
        return f"Definition for '{word}' not found in dictionary. Available words: {', '.join(data.keys())}"


@tool
def get_datetime() -> str:
    """
    Get current date and time information.
    """
    from datetime import datetime

    now = datetime.now()
    return f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')} ({now.strftime('%A, %B %d, %Y')})"


tools = [calculator, dictionary_lookup, get_datetime]


# Guardrail functions
def check_guardrails(message: str) -> Optional[str]:
    """Check for guardrail violations"""
    message_lower = message.lower()

    # Check for code execution requests
    code_keywords = [
        "exec",
        "eval",
        "import",
        "subprocess",
        "os.system",
        "shell",
        "run code",
        "execute code",
        "python code",
        "script",
        "__import__",
    ]
    if any(keyword in message_lower for keyword in code_keywords):
        if "calculator" not in message_lower and "math" not in message_lower:
            return "I cannot execute arbitrary code for security reasons."

    # Check for PII requests
    pii_patterns = [
        r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
        r"\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b",  # Credit card
        r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # Email
        r"\b\d{3}[-.]\d{3}[-.]\d{4}\b",  # Phone
    ]

    pii_keywords = [
        "social security",
        "ssn",
        "credit card",
        "password",
        "personal information",
        "phone number",
        "address",
        "email address",
        "date of birth",
        "birthday",
    ]

    if any(re.search(pattern, message) for pattern in pii_patterns):
        return "I cannot process personal identifiable information (PII) for privacy reasons."

    if any(keyword in message_lower for keyword in pii_keywords):
        if "what is" in message_lower or "define" in message_lower:
            return None  # Allow definitional questions
        return "I cannot process requests involving personal identifiable information (PII)."

    return None


def should_use_calculator(message: str) -> bool:
    """Determine if calculator tool should be used"""
    math_patterns = [
        r"\d+\s*[\+\-\*\/]\s*\d+",  # Matches basic arithmetic like "2+2", "5 * 3"
        r"\d+%\s+of\s+\d+",  # Matches percentage phrases like "17% of 350"
        r"calculate|compute|what\s+is\s+\d+",  # Matches phrases like "calculate 5", "what is 42"
        r"sqrt|sin|cos|tan|log",  # Matches math functions like "sqrt(16)", "log(10)"
        r"=|\+|\-|\*|\/|%",  # Matches math operators (even standalone)
    ]

    return any(re.search(pattern, message, re.IGNORECASE) for pattern in math_patterns)


def should_use_dictionary(message: str) -> bool:
    """Determine if dictionary tool should be used"""
    define_patterns = [
        r"define:?\s+(\w+)",
        r"what\s+does\s+(\w+)\s+mean",
        r"definition\s+of\s+(\w+)",
        r"meaning\s+of\s+(\w+)",
    ]

    return any(
        re.search(pattern, message, re.IGNORECASE) for pattern in define_patterns
    )


def should_use_datetime(message: str) -> bool:
    """Determine if datetime tool should be used"""
    time_keywords = [
        "time",
        "date",
        "today",
        "now",
        "current time",
        "current date",
        "what time is it",
        "what date is it",
        "when is it",
    ]

    return any(keyword in message.lower() for keyword in time_keywords)


def extract_word_to_define(message: str) -> str:
    """Extract the word to define from the message"""
    patterns = [
        r"define:?\s+(\w+)",
        r"what\s+does\s+(\w+)\s+mean",
        r"definition\s+of\s+(\w+)",
        r"meaning\s+of\s+(\w+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, message, re.IGNORECASE)
        if match:
            return match.group(1)
    return ""


def extract_math_expression(message: str) -> str:
    """Extract mathematical expression from the message"""
    # Look for percentage calculations first
    percentage_match = re.search(
        r"(\d+(?:\.\d+)?)%\s+of\s+(\d+(?:\.\d+)?)", message, re.IGNORECASE
    )
    if percentage_match:
        return percentage_match.group(0)

    # Look for basic arithmetic
    math_match = re.search(r"(\d+(?:\.\d+)?\s*[\+\-\*\/]\s*\d+(?:\.\d+)?)", message)
    if math_match:
        return math_match.group(1)

    # Look for expressions after "calculate" or "what is"
    calc_match = re.search(
        r"(?:calculate|what\s+is)\s+([0-9\+\-\*\/\.\s\(\)%]+)", message, re.IGNORECASE
    )
    if calc_match:
        return calc_match.group(1).strip()

    return message  # Return original if no specific pattern found


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    tool_calls_needed: bool
    guardrail_violation: Optional[str]


# Node functions for the graph
def analyze_input(state: AgentState) -> AgentState:
    """Analyze the input message and determine what tools are needed"""
    last_message = state["messages"][-1].content

    # Check guardrails first
    violation = check_guardrails(last_message)
    if violation:
        state["guardrail_violation"] = violation
        state["tool_calls_needed"] = False
        return state

    # Determine if tools are needed
    needs_calculator = should_use_calculator(last_message)
    needs_dictionary = should_use_dictionary(last_message)
    needs_datetime = should_use_datetime(last_message)

    state["tool_calls_needed"] = needs_calculator or needs_dictionary or needs_datetime
    state["guardrail_violation"] = None

    return state


def call_tools(state: AgentState) -> AgentState:
    """Call appropriate tools based on the input"""
    last_message = state["messages"][-1].content
    responses = []

    # Call calculator if needed
    if should_use_calculator(last_message):
        expression = extract_math_expression(last_message)
        result = calculator.invoke({"expression": expression})
        responses.append(result)

    # Call dictionary if needed
    if should_use_dictionary(last_message):
        word = extract_word_to_define(last_message)
        result = dictionary_lookup.invoke({"word": word})
        responses.append(result)

    # Call datetime if needed
    if should_use_datetime(last_message):
        result = get_datetime.invoke({})
        responses.append(result)

    # Add tool responses to messages
    if responses:
        tool_response = "\n".join(responses)
        state["messages"].append(AIMessage(content=tool_response))

    return state


def handle_guardrail_violation(state: AgentState) -> AgentState:
    """Handle guardrail violations"""
    violation_message = state["guardrail_violation"]
    state["messages"].append(AIMessage(content=violation_message))
    return state


def general_response(state: AgentState) -> AgentState:
    """Handle general conversation that doesn't need tools"""
    last_message = state["messages"][-1].content

    # Simple conversational responses
    response = "I'm here to help with calculations, word definitions, and time/date information. How can I assist you?"

    # Add some basic conversational patterns
    if any(greeting in last_message.lower() for greeting in ["hello", "hi", "hey"]):
        response = "Hello! I can help you with calculations, word definitions, and current date/time. What would you like to know?"
    elif any(thanks in last_message.lower() for thanks in ["thank", "thanks"]):
        response = "You're welcome! Is there anything else I can help you with?"
    elif "help" in last_message.lower():
        response = """I can help you with:
• Math calculations (e.g., "2+2", "17% of 350")
• Word definitions (e.g., "define: serendipity") 
• Current date and time (e.g., "what time is it?")

Just ask your question naturally!"""

    state["messages"].append(AIMessage(content=response))
    return state


def route_after_analysis(state: AgentState) -> str:
    """Route to appropriate node after analysis"""
    if state["guardrail_violation"]:
        return "handle_guardrail_violation"
    elif state["tool_calls_needed"]:
        return "call_tools"
    else:
        return "general_response"


# --- state ---


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0).bind_tools(tools)


def call_model(state: AgentState):
    resp = llm.invoke(state["messages"])
    return {"messages": [resp]}


# --- graph ---
graph = StateGraph(AgentState)
graph.add_node("analyze_input", analyze_input)
graph.add_node("call_tools", call_tools)
graph.add_node("handle_guardrail_violation", handle_guardrail_violation)
graph.add_node("general_response", general_response)

graph.add_edge(START, "analyze_input")
graph.add_conditional_edges(
    "analyze_input",
    route_after_analysis,
    {
        "call_tools": "call_tools",
        "handle_guardrail_violation": "handle_guardrail_violation",
        "general_response": "general_response",
    },
)
graph.add_edge("call_tools", END)
graph.add_edge("handle_guardrail_violation", END)
graph.add_edge("general_response", END)

app = graph.compile(checkpointer=InMemorySaver())
# app.get_graph().draw_png("langgraph.png")

try:
    display(Image(graph.get_graph().draw_mermaid_png()))
except Exception:
    pass


# Test function
def test_agent():
    """Test the agent with example queries"""

    test_cases = [
        "2+2",
        "17% of 350",
        "define: serendipity",
        "what time is it?",
        "hello there",
        "exec('print(hello)')",  # Should be blocked
        "what's my social security number?",  # Should be blocked
        "help me",
    ]

    print("Testing Chat Agent:")
    print("=" * 50)

    for query in test_cases:
        print(f"\nQuery: {query}")
        print("-" * 30)

        initial_state = {
            "messages": [HumanMessage(content=query)],
            "tool_calls_needed": False,
            "guardrail_violation": None,
        }

        result = app.invoke(initial_state)
        response = result["messages"][-1].content
        print(f"Response: {response}")


if __name__ == "__main__":

    # test_agent()

    # simple REPL
    thread = {"configurable": {"thread_id": "demo"}}
    while True:
        q = input("Your Input- : ")
        if q.lower() in ["exit", "bye"]:
            print("Goodbye! 👋")
            break

        initial_state = {
            "messages": [HumanMessage(content=q)],
            "tool_calls_needed": False,
            "guardrail_violation": None,
        }
        out = app.invoke(initial_state, config=thread)
        print("Agent:", out["messages"][-1].content)
