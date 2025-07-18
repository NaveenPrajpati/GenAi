from langchain_community.retrievers import WikipediaRetriever
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

# Create the Wikipedia retriever
retriever = WikipediaRetriever(top_k_results=2, lang="en")

# Run the retriever with a query
docs = retriever.invoke("Who is Elon Musk?")



# Print the retrieved results
for i, doc in enumerate(docs):
    print(f"\nResult {i}")
    print("-" * 40)
    print(doc.page_content[:500])  # print first 500 chars

