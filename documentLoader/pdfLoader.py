from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI()
loader = PyPDFLoader("/Users/naveen/Desktop/web/genAi/mongtutorial.pdf")
doc = loader.load()
print(type(doc))
print(len(doc))
print(doc[0])
print(doc[0].page_content)
print(doc[0].metadata)
