import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import (
    initialize_agent,
    AgentType,
    create_tool_calling_agent,
    Tool,
    AgentExecutor,
)
from langchain_community.utilities import SerpAPIWrapper
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# 1️⃣ Initialize the LLM
llm = ChatOpenAI()

# 2️⃣ Define tools
search = SerpAPIWrapper()  # web search
search_tool = Tool(
    name="WebSearch",
    func=search.run,
    description="Search the web for general knowledge and current events.",
)


@tool
def calculator(expression: str) -> str:
    """Compute basic mathematical operations."""
    return str(eval(expression))


tools = [search_tool, calculator]

prompt = PromptTemplate.from_template("Give simple answer of query {text}")

agent = create_tool_calling_agent(llm, tools, prompt)

agentEx = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,  # show reasoning, actions, and observations
)

# 4️⃣ Execute a query
if __name__ == "__main__":
    response = agentEx.invoke(
        {"text": "What is 15 * 7? And who is the current prime minister of Canada?"}
    )
    print(response)
