"""
Tool Binding - Connecting Tools to LLMs
========================================

LEARNING OBJECTIVES:
- Bind tools to language models using bind_tools()
- Understand tool_calls in model responses
- Execute tool calls and get results
- Handle the complete tool calling flow

CONCEPT:
Tool binding is how we tell an LLM what tools it can use. When you bind
tools to a model, the LLM can:
1. See what tools are available
2. Decide when to use them
3. Generate the proper arguments
4. Return structured tool call requests

TOOL CALLING FLOW:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Tool Calling Flow                         │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │   User: "What is 2 times 7?"                                │
    │              │                                               │
    │              ▼                                               │
    │   ┌──────────────────────┐                                  │
    │   │  LLM + Bound Tools   │                                  │
    │   │  - multiply          │                                  │
    │   │  - divide            │                                  │
    │   │  - add               │                                  │
    │   └──────────┬───────────┘                                  │
    │              │                                               │
    │              ▼                                               │
    │   AIMessage with tool_calls:                                │
    │   [{"name": "multiply",                                     │
    │     "args": {"a": 2, "b": 7}}]                              │
    │              │                                               │
    │              ▼                                               │
    │   Execute: multiply(a=2, b=7) → 14                          │
    │              │                                               │
    │              ▼                                               │
    │   Result: 14                                                │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

KEY POINTS:
1. bind_tools() doesn't execute tools - it just tells LLM what's available
2. LLM returns tool_calls in its response (structured request to call tools)
3. YOU execute the tools using the arguments from tool_calls
4. The result can be sent back to LLM for final response

PREREQUISITES:
- Completed: tools/customTools.py
- OpenAI API key in .env

NEXT STEPS:
- tools/toolbinding2.py - Advanced tool binding patterns
- langgraph/basicChatbot.py - Tools in a full agent
"""

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# STEP 1: Create Tools
# =============================================================================


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together.

    Given two integers a and b, this tool returns their product.
    Use this when you need to calculate a multiplication.
    """
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two numbers together.

    Given two integers a and b, this tool returns their sum.
    """
    return a + b


@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The name of the city

    Returns:
        A string describing the current weather.
    """
    # Simulated weather data
    weather_data = {
        "london": "Cloudy, 15°C",
        "paris": "Sunny, 22°C",
        "tokyo": "Rainy, 18°C",
        "new york": "Clear, 20°C"
    }
    return weather_data.get(city.lower(), f"Weather data for {city} not available")


# =============================================================================
# STEP 2: Create LLM and Bind Tools
# =============================================================================
# Initialize the model
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Bind tools to the model
# This tells the LLM what tools are available
llm_with_tools = llm.bind_tools([multiply, add, get_weather])

# =============================================================================
# STEP 3: Invoke and Examine Tool Calls
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("TOOL BINDING DEMO")
    print("=" * 70)

    # Test query that should trigger the multiply tool
    query = "Can you multiply 2 by 7?"

    print(f"\nQuery: {query}")
    print("-" * 70)

    # Get the model's response
    result = llm_with_tools.invoke(query)

    print(f"\nModel Response Type: {type(result).__name__}")
    print(f"Content: {result.content}")
    print(f"\nTool Calls: {result.tool_calls}")

    # ==========================================================================
    # STEP 4: Execute the Tool Call
    # ==========================================================================
    if result.tool_calls:
        print("\n" + "-" * 70)
        print("EXECUTING TOOL CALLS")
        print("-" * 70)

        for tool_call in result.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            print(f"\nTool: {tool_name}")
            print(f"Arguments: {tool_args}")

            # Execute the appropriate tool
            if tool_name == "multiply":
                tool_result = multiply.invoke(tool_args)
            elif tool_name == "add":
                tool_result = add.invoke(tool_args)
            elif tool_name == "get_weather":
                tool_result = get_weather.invoke(tool_args)
            else:
                tool_result = f"Unknown tool: {tool_name}"

            print(f"Result: {tool_result}")

    # ==========================================================================
    # STEP 5: Complete Conversation with Tool Results
    # ==========================================================================
    print("\n" + "=" * 70)
    print("COMPLETE CONVERSATION FLOW")
    print("=" * 70)

    # Build a complete conversation with tool execution
    messages = [HumanMessage(content="What is 15 times 23?")]

    # Get LLM response with tool call
    response = llm_with_tools.invoke(messages)
    messages.append(response)

    print(f"\n1. User: {messages[0].content}")
    print(f"2. LLM requests tool: {response.tool_calls}")

    # Execute tool and add result
    if response.tool_calls:
        for tool_call in response.tool_calls:
            # Execute the tool
            tool_result = multiply.invoke(tool_call["args"])

            # Create a ToolMessage with the result
            tool_message = ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            )
            messages.append(tool_message)
            print(f"3. Tool Result: {tool_result}")

        # Get final response from LLM
        final_response = llm_with_tools.invoke(messages)
        print(f"4. Final LLM Response: {final_response.content}")

    # ==========================================================================
    # STEP 6: Multiple Tool Calls
    # ==========================================================================
    print("\n" + "=" * 70)
    print("MULTIPLE TOOL CALLS")
    print("=" * 70)

    # Query that might trigger multiple tools
    query = "What is 5 + 3, and also what is 4 * 6?"
    print(f"\nQuery: {query}")

    response = llm_with_tools.invoke(query)

    if response.tool_calls:
        print(f"\nLLM requested {len(response.tool_calls)} tool calls:")
        for i, tc in enumerate(response.tool_calls, 1):
            print(f"  {i}. {tc['name']}({tc['args']})")

            # Execute each tool
            if tc["name"] == "multiply":
                result = multiply.invoke(tc["args"])
            elif tc["name"] == "add":
                result = add.invoke(tc["args"])
            else:
                result = "Unknown"

            print(f"     Result: {result}")

    # ==========================================================================
    # BEST PRACTICES
    # ==========================================================================
    print("\n" + "=" * 70)
    print("BEST PRACTICES")
    print("=" * 70)
    print("""
    1. Always check if tool_calls exists before processing
    2. Handle unknown tool names gracefully
    3. Create ToolMessage with correct tool_call_id for proper tracking
    4. Send tool results back to LLM for natural language response
    5. Consider error handling when tools fail
    6. Use temperature=0 for more consistent tool selection
    7. Keep tool descriptions clear - they guide the LLM's choices
    """)
