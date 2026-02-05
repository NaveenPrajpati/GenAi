import os
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()
# 1. Setup Model (Ensure your API keys are set in environment variables)
# In v1.x, models are more 'agent-aware' by default
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# 2. Define Tools
# We'll use Tavily for search, which is the standard for LangChain v1.x
search_tool = TavilySearch(
    max_results=2,
    topic="general",
)
tools = [search_tool]

# 3. Initialize the Agent
# This single function replaces the old 'Chain + Executor' setup.
# It internally constructs a LangGraph with a tool-calling loop.
agent = create_agent(
    model=llm,
    tools=tools,
    system_prompt="""You are a research assistant. 
    Use the search tool to find up-to-date information. 
    Always cite your sources briefly.""",
)

# 4. Invoke the Agent
# The input schema is now standardized: a list of messages or a dict.
query = "What are the key features of the LangChain v1.2 release from early 2026?"

print("--- Agent is processing ---")
for step in agent.stream(
    {"messages": query},
    stream_mode="values",
):
    step["messages"][-1].pretty_print()
