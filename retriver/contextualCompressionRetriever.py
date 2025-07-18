from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from dotenv import load_dotenv
load_dotenv()
# 1️⃣ Load & chunk your source text
docs = TextLoader("state_of_the_union.txt").load()
splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
chunks = splitter.split_documents(docs)

# 2️⃣ Create vectorstore & base retriever
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(chunks, embeddings)
base_retriever = vectorstore.as_retriever()

# 3️⃣ Initialize LLM compressor
llm = OpenAI(temperature=0)
compressor = LLMChainExtractor.from_llm(llm)

# 4️⃣ Build contextual compression retriever
compression_retriever = ContextualCompressionRetriever(
    base_retriever=base_retriever,
    base_compressor=compressor
)

# 5️⃣ Query!
query = "What did the president say about Justice Breyer?"
docs = compression_retriever.get_relevant_documents(query)

# 6️⃣ Show compressed results:
for i, d in enumerate(docs):
    print(f"--- Compressed Doc {i+1} ---")
    print(d.page_content)