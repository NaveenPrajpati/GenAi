import os
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_openai.chat_models import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor

from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from langchain import hub

load_dotenv()

llm = ChatOpenAI()

search = TavilySearch()


@tool
def calculator(expression: str) -> str:
    """Compute basic mathematical operations."""
    return str(eval(expression))


tools = [search, calculator]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a helpful assistant with access to web search and calculator tools.
    Please provide accurate and helpful responses. Use the calculator for math problems 
    and search for current information when needed.""",
        ),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)

promt1 = hub.pull("hwchase17/openai-functions-agent")

# Popular LangChain Hub Prompts for Agents:
# "hwchase17/openai-functions-agent" - For OpenAI function calling (recommended)
# "hwchase17/react" - For ReAct (Reasoning + Acting) agents
# "hwchase17/react-chat" - Chat version of ReAct
# "hwchase17/structured-chat-agent" - For structured conversations


agent = create_tool_calling_agent(llm, tools, prompt)

agentEx = AgentExecutor(
    agent=agent,
    tools=tools,
    handle_parsing_errors=True,
    verbose=True,  # show reasoning, actions, and observations
)

if __name__ == "__main__":
    response = agentEx.invoke(
        {"input": "What is 15 * 7? And who is the current prime minister of Canada?"}
    )
    print(response)
