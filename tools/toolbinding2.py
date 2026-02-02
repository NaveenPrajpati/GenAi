"""
Advanced Tool Binding - Patterns and Best Practices
=====================================================

LEARNING OBJECTIVES:
- Create tools with complex input schemas
- Build tools that call external APIs
- Implement tools with logging and callbacks
- Create composite tools (tools that use other tools)
- Handle tool errors gracefully

CONCEPT:
This tutorial covers advanced tool patterns you'll encounter in
production applications:

ADVANCED PATTERNS:
    ┌─────────────────────────────────────────────────────────────┐
    │                  Advanced Tool Patterns                      │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  1. Structured Input Tools                                  │
    │     - Complex Pydantic schemas                              │
    │     - Nested objects                                        │
    │     - Optional fields with defaults                         │
    │                                                              │
    │  2. API Integration Tools                                   │
    │     - HTTP requests                                         │
    │     - Error handling                                        │
    │     - Rate limiting                                         │
    │                                                              │
    │  3. Logging Tools                                           │
    │     - Audit trails                                          │
    │     - Performance monitoring                                │
    │     - Debug output                                          │
    │                                                              │
    │  4. Composite Tools                                         │
    │     - Tools calling tools                                   │
    │     - Multi-step operations                                 │
    │     - Workflow orchestration                                │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

PREREQUISITES:
- Completed: tools/customTools.py, tools/toolbinding.py
- Understanding of Pydantic models
- OpenAI API key in .env

NEXT STEPS:
- langgraph/basicChatbot.py - Tools in a full agent
- practicsProjects/1tool-use.py - Production tool patterns
"""

from langchain_core.tools import tool, BaseTool, StructuredTool
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from typing import Optional, List, Literal
from datetime import datetime
from dotenv import load_dotenv
import json
import requests

load_dotenv()

# =============================================================================
# PATTERN 1: Complex Pydantic Schemas
# =============================================================================


class SearchFilters(BaseModel):
    """Nested schema for search filters."""
    category: Optional[str] = Field(None, description="Category to filter by")
    min_price: Optional[float] = Field(None, description="Minimum price")
    max_price: Optional[float] = Field(None, description="Maximum price")
    in_stock: bool = Field(True, description="Only show in-stock items")


class ProductSearchInput(BaseModel):
    """Schema for product search with complex nested structure."""
    query: str = Field(description="The search query string")
    filters: Optional[SearchFilters] = Field(None, description="Optional search filters")
    sort_by: Literal["price", "rating", "relevance"] = Field(
        default="relevance",
        description="How to sort results"
    )
    limit: int = Field(default=10, ge=1, le=100, description="Max results to return")


@tool(args_schema=ProductSearchInput)
def search_products(
    query: str,
    filters: Optional[SearchFilters] = None,
    sort_by: str = "relevance",
    limit: int = 10
) -> str:
    """Search for products in the catalog.

    Use this tool when you need to find products based on a search query.
    You can optionally filter by category, price range, and stock status.
    """
    # Simulated search logic
    results = {
        "query": query,
        "sort_by": sort_by,
        "limit": limit,
        "filters_applied": filters.model_dump() if filters else None,
        "results": [
            {"name": f"Product {i}", "price": 10.0 * i, "rating": 4.5}
            for i in range(1, min(limit + 1, 4))
        ]
    }
    return json.dumps(results, indent=2)


# =============================================================================
# PATTERN 2: API Integration Tool
# =============================================================================


class WeatherInput(BaseModel):
    """Input schema for weather lookup."""
    location: str = Field(description="City and country, e.g., 'London, UK'")


class WeatherTool(BaseTool):
    """Tool for fetching weather data from an API.

    Demonstrates:
    - HTTP requests with error handling
    - Timeout configuration
    - Response parsing
    """
    name: str = "weather_lookup"
    description: str = "Get current weather information for a specific location"
    args_schema: type[BaseModel] = WeatherInput

    # Configuration
    timeout_seconds: int = 5

    def _run(self, location: str) -> str:
        """Fetch weather for a location (mock implementation)."""
        # In production, you would call a real API:
        # try:
        #     response = requests.get(
        #         f"https://api.weather.com/v1/current",
        #         params={"location": location},
        #         timeout=self.timeout_seconds,
        #         headers={"User-Agent": "WeatherBot/1.0"}
        #     )
        #     response.raise_for_status()
        #     data = response.json()
        #     return json.dumps(data)
        # except requests.exceptions.Timeout:
        #     return "Error: Weather API request timed out"
        # except requests.exceptions.RequestException as e:
        #     return f"Error fetching weather: {str(e)}"

        # Mock implementation
        mock_data = {
            "london, uk": {"temp": 18, "condition": "Cloudy", "humidity": 75},
            "new york, us": {"temp": 22, "condition": "Sunny", "humidity": 45},
            "tokyo, japan": {"temp": 16, "condition": "Rainy", "humidity": 90},
        }

        location_lower = location.lower()
        if location_lower in mock_data:
            data = mock_data[location_lower]
            return (
                f"Weather in {location}: {data['condition']}, "
                f"{data['temp']}°C, Humidity: {data['humidity']}%"
            )
        return f"Weather data not available for {location}"

    async def _arun(self, location: str) -> str:
        """Async version - could use aiohttp for true async."""
        return self._run(location)


# =============================================================================
# PATTERN 3: Logging and Monitoring Tool
# =============================================================================


class LoggingCalculator(BaseTool):
    """Calculator that logs all operations.

    Demonstrates:
    - Operation logging
    - Audit trails
    - Performance monitoring
    """
    name: str = "logging_calculator"
    description: str = "Perform calculations with full logging"

    # Internal state
    operation_log: List[dict] = []

    def _run(self, expression: str) -> str:
        """Execute calculation with logging."""
        start_time = datetime.now()

        try:
            # Safe evaluation (in production, use a proper math parser)
            # NEVER use eval with untrusted input in production!
            allowed_names = {"abs": abs, "round": round, "min": min, "max": max}
            result = eval(expression, {"__builtins__": {}}, allowed_names)

            end_time = datetime.now()
            duration_ms = (end_time - start_time).total_seconds() * 1000

            # Log the operation
            log_entry = {
                "timestamp": start_time.isoformat(),
                "expression": expression,
                "result": result,
                "status": "success",
                "duration_ms": duration_ms
            }
            self.operation_log.append(log_entry)

            return f"Result: {result} (logged in {duration_ms:.2f}ms)"

        except Exception as e:
            # Log errors too
            log_entry = {
                "timestamp": start_time.isoformat(),
                "expression": expression,
                "error": str(e),
                "status": "error"
            }
            self.operation_log.append(log_entry)

            return f"Calculation error: {str(e)}"

    def get_operation_history(self) -> List[dict]:
        """Get the full operation log."""
        return self.operation_log

    async def _arun(self, expression: str) -> str:
        return self._run(expression)


# =============================================================================
# PATTERN 4: Composite Tool (Tool that uses other tools)
# =============================================================================


class DataAnalysisTool(BaseTool):
    """Complex tool that combines multiple operations.

    Demonstrates:
    - Tools calling other tools
    - Multi-step workflows
    - Result aggregation
    """
    name: str = "data_analyzer"
    description: str = (
        "Perform complex data analysis by fetching data and calculating statistics. "
        "Use this for comprehensive analysis tasks."
    )

    # Inject dependencies
    calculator: LoggingCalculator = None
    weather_tool: WeatherTool = None

    def __init__(self, **data):
        super().__init__(**data)
        self.calculator = LoggingCalculator()
        self.weather_tool = WeatherTool()

    def _run(self, query: str) -> str:
        """Perform multi-step analysis."""
        results = []

        # Example: Query could be "analyze sales with weather correlation"
        if "weather" in query.lower():
            # Get weather data for multiple cities
            cities = ["London, UK", "New York, US", "Tokyo, Japan"]
            weather_data = []

            for city in cities:
                weather = self.weather_tool._run(city)
                weather_data.append(f"  - {weather}")

            results.append("Weather Analysis:")
            results.extend(weather_data)

        if "calculate" in query.lower() or "average" in query.lower():
            # Perform some calculations
            calc_result = self.calculator._run("(10 + 20 + 30) / 3")
            results.append(f"\nCalculation: {calc_result}")

        if not results:
            return (
                "Analysis complete. Specify 'weather' for weather data "
                "or 'calculate' for numerical analysis."
            )

        return "\n".join(results)

    async def _arun(self, query: str) -> str:
        return self._run(query)


# =============================================================================
# PATTERN 5: URL Analyzer with Error Handling
# =============================================================================


class URLAnalyzerTool(BaseTool):
    """Analyze a URL and extract information.

    Demonstrates:
    - Real HTTP requests
    - Comprehensive error handling
    - Content parsing
    """
    name: str = "url_analyzer"
    description: str = "Analyze a URL and extract basic information about the webpage"

    def _run(self, url: str) -> str:
        """Analyze a URL and return basic information."""
        try:
            response = requests.get(
                url,
                timeout=5,
                headers={"User-Agent": "Mozilla/5.0 (compatible; URLAnalyzer/1.0)"}
            )

            info = {
                "url": url,
                "status_code": response.status_code,
                "content_length": len(response.content),
                "content_type": response.headers.get("content-type", "Unknown"),
                "title": "Could not extract title"
            }

            # Try to extract title from HTML
            if "text/html" in response.headers.get("content-type", ""):
                import re
                title_match = re.search(
                    r"<title>(.*?)</title>",
                    response.text,
                    re.IGNORECASE | re.DOTALL
                )
                if title_match:
                    info["title"] = title_match.group(1).strip()[:100]

            return json.dumps(info, indent=2)

        except requests.exceptions.Timeout:
            return json.dumps({"error": "Request timed out", "url": url})
        except requests.exceptions.ConnectionError:
            return json.dumps({"error": "Could not connect to URL", "url": url})
        except requests.exceptions.RequestException as e:
            return json.dumps({"error": str(e), "url": url})

    async def _arun(self, url: str) -> str:
        return self._run(url)


# =============================================================================
# DEMO: Using Advanced Tools with LLM
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("ADVANCED TOOL BINDING DEMO")
    print("=" * 70)

    # Initialize tools
    weather_tool = WeatherTool()
    logging_calc = LoggingCalculator()
    data_analyzer = DataAnalysisTool()
    url_analyzer = URLAnalyzerTool()

    # Create list of tools
    tools = [
        search_products,
        weather_tool,
        logging_calc,
        data_analyzer,
        url_analyzer
    ]

    # Bind to LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    llm_with_tools = llm.bind_tools(tools)

    # Test queries
    test_queries = [
        "Search for laptops under $1000",
        "What's the weather in Tokyo, Japan?",
        "Calculate 15 * 23 + 100",
        "Analyze the weather patterns across major cities",
    ]

    for query in test_queries:
        print(f"\n{'='*70}")
        print(f"Query: {query}")
        print("-" * 70)

        response = llm_with_tools.invoke(query)

        if response.tool_calls:
            for tc in response.tool_calls:
                print(f"Tool: {tc['name']}")
                print(f"Args: {json.dumps(tc['args'], indent=2)}")

                # Execute the tool
                tool_map = {
                    "search_products": search_products,
                    "weather_lookup": weather_tool,
                    "logging_calculator": logging_calc,
                    "data_analyzer": data_analyzer,
                    "url_analyzer": url_analyzer
                }

                if tc["name"] in tool_map:
                    result = tool_map[tc["name"]].invoke(tc["args"])
                    print(f"Result: {result}")
        else:
            print(f"LLM Response: {response.content}")

    # Show operation history from logging calculator
    print("\n" + "=" * 70)
    print("LOGGING CALCULATOR HISTORY")
    print("=" * 70)
    for entry in logging_calc.get_operation_history():
        print(f"  {entry['timestamp']}: {entry.get('expression', 'N/A')} "
              f"= {entry.get('result', entry.get('error', 'N/A'))}")

    print("\n" + "=" * 70)
    print("BEST PRACTICES FOR PRODUCTION TOOLS")
    print("=" * 70)
    print("""
    1. ALWAYS validate input with Pydantic schemas
    2. Use timeouts for external API calls
    3. Implement comprehensive error handling
    4. Log all operations for debugging and auditing
    5. Keep tools focused - one responsibility per tool
    6. Document all possible error conditions
    7. Consider rate limiting for expensive operations
    8. Use async when dealing with I/O operations
    9. Never use eval() with untrusted input
    10. Return structured data (JSON) for complex results
    """)
