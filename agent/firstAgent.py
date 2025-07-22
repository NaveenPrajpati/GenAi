from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain.schema.runnable import RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain.agents import create_react_agent , AgentExecutor
from langchain import hub
from dotenv import load_dotenv
import os
import requests
from langchain_community.tools import DuckDuckGoSearchRun

load_dotenv()

search_tool=DuckDuckGoSearchRun()
result=search_tool.invoke('top news in noida today')
print(result)

@tool
def findWeather(city:str):
    "function to fetch weather of given city"
    response=requests.get(f' urkl {city}')
    return response.json()

llm=ChatOpenAI()
llm.invoke('hi')

prompt = hub.pull('hwchase17/react')

agent= create_react_agent(
    llm=llm,
    tools=[search_tool,findWeather],
    prompt=prompt
)

agent_excutor=AgentExecutor(
    agent=agent,
    tools=[search_tool,findWeather],
    verbose=True
)

response =agent_excutor.invoke({'input':' find current weather of capital of utter pradesh'})