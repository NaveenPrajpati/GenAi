"""
Custom Tool Binding Example in LangChain
This example demonstrates how to create custom tools and bind them to LLMs
"""

from langchain.tools import BaseTool
from langchain.agents import initialize_agent, AgentType
from langchain_community.llms import OpenAI
from langchain_openai import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.tools.base import StructuredTool
from pydantic import BaseModel, Field
from typing import Optional, Type
from dotenv import load_dotenv
import requests

import json
import math

load_dotenv()
# Method 1: Creating a custom tool by inheriting from BaseTool
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "Useful for performing mathematical calculations. Input should be a mathematical expression."
    
    def _run(self, query: str) -> str:
        """Execute the calculator tool."""
        try:
            # Safe evaluation of mathematical expressions
            result = eval(query.replace("^", "**"))
            return f"The result of {query} is {result}"
        except Exception as e:
            return f"Error in calculation: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        """Async version of _run method."""
        return self._run(query)


# Method 2: Using Pydantic models for structured inputs
class WeatherInput(BaseModel):
    location: str = Field(description="The city and country, e.g., 'London, UK'")

class WeatherTool(BaseTool):
    name = "weather_lookup"
    description = "Get current weather information for a specific location"
    args_schema: Type[BaseModel] = WeatherInput
    
    def _run(self, location: str) -> str:
        """Get weather for a location (mock implementation)."""
        # In a real implementation, you'd call a weather API
        mock_weather_data = {
            "London, UK": "Cloudy, 18°C",
            "New York, US": "Sunny, 22°C", 
            "Tokyo, Japan": "Rainy, 16°C"
        }
        
        weather = mock_weather_data.get(location, "Weather data not available")
        return f"Weather in {location}: {weather}"
    
    async def _arun(self, location: str) -> str:
        return self._run(location)


# Method 3: Creating tools using StructuredTool.from_function
def search_database(query: str, table: str = "users") -> str:
    """
    Search a database table for information.
    
    Args:
        query: The search query
        table: The table to search in (default: users)
    """
    # Mock database search
    mock_results = {
        "users": {"john": "John Doe, Software Engineer", "jane": "Jane Smith, Data Scientist"},
        "products": {"laptop": "MacBook Pro, $2000", "phone": "iPhone 14, $999"}
    }
    
    table_data = mock_results.get(table, {})
    results = [v for k, v in table_data.items() if query.lower() in k.lower()]
    
    if results:
        return f"Found {len(results)} results: {', '.join(results)}"
    return f"No results found for '{query}' in table '{table}'"

# Create the structured tool
database_tool = StructuredTool.from_function(
    func=search_database,
    name="database_search",
    description="Search database tables for information. Useful for finding user or product data."
)


# Method 4: Custom tool with complex logic and external API calls
class URLAnalyzerTool(BaseTool):
    name = "url_analyzer"
    description = "Analyze a URL and extract basic information about the webpage"
    
    def _run(self, url: str) -> str:
        """Analyze a URL and return basic information."""
        try:
            response = requests.get(url, timeout=5, headers={'User-Agent': 'Mozilla/5.0'})
            
            info = {
                "status_code": response.status_code,
                "content_length": len(response.content),
                "content_type": response.headers.get('content-type', 'Unknown'),
                "title": "Could not extract title"
            }
            
            # Try to extract title
            if 'text/html' in response.headers.get('content-type', ''):
                import re
                title_match = re.search(r'<title>(.*?)</title>', response.text, re.IGNORECASE)
                if title_match:
                    info["title"] = title_match.group(1).strip()
            
            return json.dumps(info, indent=2)
            
        except Exception as e:
            return f"Error analyzing URL: {str(e)}"
    
    async def _arun(self, url: str) -> str:
        return self._run(url)


# Method 5: Tool with return_direct=True (returns result directly to user)
class TimeTool(BaseTool):
    name = "current_time"
    description = "Get the current time and date"
    return_direct = True  # This will return the result directly without further processing
    
    def _run(self, query: str = "") -> str:
        """Get current time."""
        from datetime import datetime
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"Current time: {current_time}"
    
    async def _arun(self, query: str = "") -> str:
        return self._run(query)


# Example usage and agent setup
def main():
    # Initialize the language model
    llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo")
    
    # Create list of custom tools
    tools = [
        CalculatorTool(),
        WeatherTool(),
        database_tool,
        URLAnalyzerTool(),
        TimeTool()
    ]
    
    # Method 1: Using tools with an agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        handle_parsing_errors=True
    )
    
    # Example queries
    example_queries = [
        "What's 25 * 34 + 100?",
        "What's the weather like in London, UK?",
        "Search for john in the users table",
        "What time is it?",
        "Analyze the URL https://www.example.com"
    ]
    
    print("=== Custom Tool Binding Demo ===\n")
    
    for query in example_queries:
        print(f"Query: {query}")
        try:
            result = agent.run(query)
            print(f"Result: {result}\n")
        except Exception as e:
            print(f"Error: {str(e)}\n")
    
    # Method 2: Direct tool binding to LLM (for newer versions)
    # This is useful when you want more control over tool execution
    print("=== Direct Tool Binding ===\n")
    
    # Bind tools directly to the model
    llm_with_tools = llm.bind_tools(tools)
    
    # Example of invoking with tool binding
    response = llm_with_tools.invoke("Calculate 15 * 20")
    print(f"Direct binding result: {response}")


# Method 6: Custom tool with callbacks and logging
class LoggingCalculatorTool(BaseTool):
    name = "logging_calculator"
    description = "Calculator with logging capabilities"
    
    def __init__(self, log_file: str = "calculations.log"):
        super().__init__()
        self.log_file = log_file
    
    def _run(self, query: str) -> str:
        """Execute calculation with logging."""
        try:
            result = eval(query.replace("^", "**"))
            
            # Log the calculation
            import datetime
            log_entry = f"{datetime.datetime.now()}: {query} = {result}\n"
            
            with open(self.log_file, "a") as f:
                f.write(log_entry)
            
            return f"Calculated: {query} = {result} (logged to {self.log_file})"
            
        except Exception as e:
            return f"Calculation error: {str(e)}"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)


# Method 7: Tool that can call other tools (composite tool)
class DataAnalysisTool(BaseTool):
    name = "data_analysis"
    description = "Perform complex data analysis by combining multiple operations"
    
    def __init__(self, calculator_tool, database_tool):
        super().__init__()
        self.calculator = calculator_tool
        self.database = database_tool
    
    def _run(self, query: str) -> str:
        """Perform data analysis."""
        # This is a simple example - in practice, this could be much more complex
        if "average" in query.lower():
            # Mock: get some data and calculate average
            data_points = [10, 20, 30, 40, 50]  # This could come from database_tool
            average = sum(data_points) / len(data_points)
            calculation_result = self.calculator._run(f"sum([10,20,30,40,50])/5")
            
            return f"Data analysis result: Average = {average}. Calculator verification: {calculation_result}"
        
        return "Data analysis complete - please specify what type of analysis you need"
    
    async def _arun(self, query: str) -> str:
        return self._run(query)


if __name__ == "__main__":
    # Note: You'll need to set your OpenAI API key as an environment variable
    # export OPENAI_API_KEY="your-api-key-here"
    
    print("Custom Tool Binding Example")
    print("Make sure to set your OPENAI_API_KEY environment variable before running")
    
    # Uncomment the line below to run the main demo
    # main()