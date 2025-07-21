from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import requests

load_dotenv()

@tool
def multiply(a: int, b: int) -> int:
  """Given 2 numbers a and b this tool returns their product"""
  return a * b

print(multiply.invoke({'a':3, 'b':4}))

llm = ChatOpenAI()
llmWithTool = llm.bind_tools([multiply])

result= llmWithTool.invoke('can you multiply 2 into 7')

result.tool_calls[0]['args']

multiply.invoke(result.tool_calls[0]['args'])
multiply.invoke(result.tool_calls[0])