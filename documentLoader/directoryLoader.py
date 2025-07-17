from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

llm=ChatOpenAI()
loader =DirectoryLoader(path='forlderpath',glob='*.pdf',loader_cls=PyPDFLoader)
# doc=loader.load()
doc=loader.lazy_load()
# print(doc)
print(doc[0].page_content)
print(doc[0].metadata)