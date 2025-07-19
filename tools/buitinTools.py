from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ShellTool

searchTool=DuckDuckGoSearchRun()
result=searchTool.invoke('new 7 seater cars')

print(result)