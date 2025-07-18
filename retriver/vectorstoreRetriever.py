from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever 
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()

# Load & split documents
loader = TextLoader("my_text.txt",encoding='utf-8')
docs = loader.load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = splitter.split_documents(docs)

# Embed & build vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings,collection_name='new_collection')

# Instantiate retriever

# this is simple retriever
# retriever = vectorstore.as_retriever(search_kwargs={"k": 2}) 

#this enable mmr(maximum margin retriever)
# retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={"k": 2})

#this is multi query retriever 
retriever = MultiQueryRetriever.from_llm(
    retriever= vectorstore.as_retriever(search_type='similarity', search_kwargs={"k": 2}),
    llm=ChatOpenAI()
)


# Query
relevant_docs = retriever.invoke("What does the text say about climate change?")
for d in relevant_docs:
    print(d.page_content[:200], "…")