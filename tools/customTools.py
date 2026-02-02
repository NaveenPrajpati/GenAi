"""
Custom Tools - Three Ways to Create Tools in LangChain
=======================================================

LEARNING OBJECTIVES:
- Create tools using the @tool decorator (simplest)
- Create tools using StructuredTool (more control)
- Create tools by subclassing BaseTool (most flexible)
- Understand tool schemas and how LLMs use them

CONCEPT:
Tools are functions that LLMs can call to interact with the outside world.
They enable agents to:
- Search the web
- Execute code
- Query databases
- Call APIs
- Perform calculations

TOOL ANATOMY:
Every tool has these key components:
    ┌─────────────────────────────────────────────────────────────┐
    │                        Tool Structure                        │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │   name: str          →  Identifier for the LLM              │
    │   description: str   →  Explains WHEN to use the tool       │
    │   args_schema: dict  →  Defines input parameters            │
    │   func: callable     →  The actual function to execute      │
    │                                                              │
    │   Example:                                                   │
    │   name: "calculator"                                        │
    │   description: "Useful for math calculations"               │
    │   args: {"a": int, "b": int}                                │
    │   func: lambda a, b: a * b                                  │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

THREE APPROACHES:

1. @tool Decorator (Simple):
   - Best for quick, simple tools
   - Automatically extracts schema from function signature
   - Uses docstring as description

2. StructuredTool (Intermediate):
   - More control over metadata
   - Explicit Pydantic schema
   - Good for tools with complex inputs

3. BaseTool Subclass (Advanced):
   - Full control over behavior
   - Supports async execution
   - Custom callback handling

WHEN TO USE EACH:
┌─────────────────┬───────────────────────────────────────────┐
│ Approach        │ Use When                                  │
├─────────────────┼───────────────────────────────────────────┤
│ @tool           │ Simple functions, quick prototyping       │
│ StructuredTool  │ Need custom name/description, validation  │
│ BaseTool        │ Complex logic, async, custom callbacks    │
└─────────────────┴───────────────────────────────────────────┘

PREREQUISITES:
- Understanding of Python functions and type hints
- Basic knowledge of Pydantic models

NEXT STEPS:
- tools/toolbinding.py - Bind tools to LLMs
- practicsProjects/1tool-use.py - Tools in agents
"""

from langchain_core.tools import tool, StructuredTool, BaseTool
from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from pydantic import BaseModel, Field
from typing import Optional

# =============================================================================
# APPROACH 1: @tool Decorator (Simplest)
# =============================================================================
# The @tool decorator is the quickest way to create a tool.
# It automatically:
# - Uses function name as tool name
# - Uses docstring as description
# - Infers args_schema from type hints


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together.

    Use this tool when you need to calculate the product of two integers.
    """
    return a * b


# Let's inspect what the decorator created:
print("=" * 70)
print("APPROACH 1: @tool Decorator")
print("=" * 70)
print(f"Name:        {multiply.name}")
print(f"Description: {multiply.description}")
print(f"Args:        {multiply.args}")
print(f"Schema:      {multiply.args_schema.model_json_schema()}")
print(f"\nInvoke test: multiply(2, 4) = {multiply.invoke({'a': 2, 'b': 4})}")


# More examples with @tool
@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for information.

    Args:
        query: The search query
        max_results: Maximum number of results to return (default 5)

    Use this when you need to find current information from the internet.
    """
    # Simulated search
    return f"Found {max_results} results for: {query}"


@tool
def get_weather(city: str, units: str = "celsius") -> str:
    """Get the current weather for a city.

    Args:
        city: The city name
        units: Temperature units - 'celsius' or 'fahrenheit'
    """
    # Simulated weather API
    return f"Weather in {city}: 22 degrees {units}, sunny"


# =============================================================================
# APPROACH 2: StructuredTool (More Control)
# =============================================================================
# StructuredTool gives you explicit control over the tool's metadata
# and uses a Pydantic model for input validation.


class CalculatorInput(BaseModel):
    """Input schema for calculator tool."""
    a: int = Field(description="The first number")
    b: int = Field(description="The second number")


def multiply_func(a: int, b: int) -> int:
    """Multiply two numbers."""
    return a * b


calculator = StructuredTool.from_function(
    func=multiply_func,
    name="Calculator",
    description="Multiply two numbers together. Use this for multiplication calculations.",
    args_schema=CalculatorInput,
    return_direct=True,  # Return result directly without LLM processing
)

print("\n" + "=" * 70)
print("APPROACH 2: StructuredTool")
print("=" * 70)
print(f"Name:        {calculator.name}")
print(f"Description: {calculator.description}")
print(f"Args:        {calculator.args}")
print(f"\nInvoke test: Calculator(2, 3) = {calculator.invoke({'a': 2, 'b': 3})}")


# StructuredTool with async support
async def async_multiply(a: int, b: int) -> int:
    """Async version of multiply."""
    import asyncio
    await asyncio.sleep(0.1)  # Simulate async operation
    return a * b


calculator_async = StructuredTool.from_function(
    func=multiply_func,
    coroutine=async_multiply,  # Provide async version
    name="AsyncCalculator",
    description="Async multiplication calculator",
    args_schema=CalculatorInput,
)


# =============================================================================
# APPROACH 3: BaseTool Subclass (Most Flexible)
# =============================================================================
# Subclassing BaseTool gives you complete control:
# - Custom initialization logic
# - Custom callback handling
# - Complex async implementations
# - State management


class AdvancedCalculatorInput(BaseModel):
    """Input schema for the advanced calculator."""
    a: int = Field(description="First number in the calculation")
    b: int = Field(description="Second number in the calculation")


class CustomCalculatorTool(BaseTool):
    """A custom calculator tool with full control over behavior.

    This approach is best when you need:
    - Custom initialization
    - Complex async handling
    - Callback management
    - Internal state
    """

    # Class attributes define the tool metadata
    name: str = "AdvancedCalculator"
    description: str = (
        "An advanced calculator for multiplication. "
        "Useful when you need to multiply two integers together."
    )
    args_schema: type[BaseModel] = AdvancedCalculatorInput
    return_direct: bool = True

    # Optional: track usage
    call_count: int = 0

    def _run(
        self,
        a: int,
        b: int,
        run_manager: Optional[CallbackManagerForToolRun] = None
    ) -> int:
        """Execute the tool synchronously.

        This is the main method that implements the tool's logic.
        The run_manager can be used to emit callbacks during execution.
        """
        self.call_count += 1

        # Optionally emit a callback
        if run_manager:
            run_manager.on_text(f"Calculating {a} * {b}...")

        return a * b

    async def _arun(
        self,
        a: int,
        b: int,
        run_manager: Optional[AsyncCallbackManagerForToolRun] = None
    ) -> int:
        """Execute the tool asynchronously.

        For CPU-bound operations, you can delegate to the sync version.
        For I/O-bound operations (API calls, DB queries), implement
        true async logic here.
        """
        # For simple operations, delegate to sync version
        # For I/O operations, implement true async logic
        return self._run(a, b, run_manager=run_manager.get_sync() if run_manager else None)


# Create an instance
advanced_calculator = CustomCalculatorTool()

print("\n" + "=" * 70)
print("APPROACH 3: BaseTool Subclass")
print("=" * 70)
print(f"Name:         {advanced_calculator.name}")
print(f"Description:  {advanced_calculator.description}")
print(f"Args:         {advanced_calculator.args}")
print(f"Return Direct:{advanced_calculator.return_direct}")
print(f"\nInvoke test: AdvancedCalculator(5, 3) = {advanced_calculator.invoke({'a': 5, 'b': 3})}")


# =============================================================================
# COMPARISON: All Three Approaches
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("COMPARISON: All Three Approaches")
    print("=" * 70)

    tools = [
        ("@tool decorator", multiply),
        ("StructuredTool", calculator),
        ("BaseTool subclass", advanced_calculator)
    ]

    for name, tool_instance in tools:
        print(f"\n{name}:")
        print(f"  Name: {tool_instance.name}")
        print(f"  Description: {tool_instance.description[:50]}...")
        print(f"  Args: {list(tool_instance.args.keys())}")
        result = tool_instance.invoke({"a": 7, "b": 8})
        print(f"  7 * 8 = {result}")

    # ==========================================================================
    # BEST PRACTICES
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BEST PRACTICES")
    print("=" * 70)
    print("""
    1. Write CLEAR descriptions - LLMs use these to decide when to call tools
    2. Use DESCRIPTIVE argument names - 'query' not 'q', 'city_name' not 'c'
    3. Add FIELD descriptions in Pydantic schemas for complex inputs
    4. Handle ERRORS gracefully - return error messages, don't raise exceptions
    5. Keep tools FOCUSED - one tool, one purpose
    6. Use TYPE HINTS - they become the input schema
    7. Document SIDE EFFECTS - if a tool modifies state, say so
    8. Consider RETURN FORMAT - what format is most useful for the LLM?
    """)

    # ==========================================================================
    # ADVANCED: Tool with error handling
    # ==========================================================================
    @tool
    def divide(a: float, b: float) -> str:
        """Divide two numbers.

        Returns the result of a divided by b.
        Handles division by zero gracefully.
        """
        if b == 0:
            return "Error: Cannot divide by zero"
        return str(a / b)

    print("\n" + "=" * 70)
    print("ADVANCED: Tool with Error Handling")
    print("=" * 70)
    print(f"divide(10, 2) = {divide.invoke({'a': 10, 'b': 2})}")
    print(f"divide(10, 0) = {divide.invoke({'a': 10, 'b': 0})}")
