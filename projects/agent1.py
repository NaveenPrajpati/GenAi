from langchain_tavily import TavilySearch
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain import hub
from langchain.agents import create_tool_calling_agent, AgentExecutor
import os

# Load environment variables
load_dotenv()


def main():
    try:
        # Initialize search tool with proper class name
        search = TavilySearch(
            max_results=2,
            topic="general",
        )

        # Test search functionality
        print("Testing search tool...")
        result = search.invoke("what is the weather in delhi")
        print(f"Search result: {result}")

        # Load and process documents
        print("\nLoading and processing documents...")
        loader = WebBaseLoader(
            "https://www.geeksforgeeks.org/mongodb/mongodb-an-introduction/"
        )
        docs = loader.load()

        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        documents = text_splitter.split_documents(docs)

        # Create vector store and retriever
        vector_store = FAISS.from_documents(documents, OpenAIEmbeddings())
        retriever = vector_store.as_retriever()

        # Test retriever
        print("Testing retriever...")
        test_docs = retriever.invoke("why use mongo db")
        if test_docs:
            print(f"Retrieved document preview: {test_docs[0].page_content[:200]}...")

        # Create retriever tool
        retriever_tool = create_retriever_tool(
            retriever,
            "mongo_search",
            "Search for information about mongo db. For any questions about Mongo db, you must use this tool!",
        )

        # Combine tools
        tools = [search, retriever_tool]

        # Initialize model with proper import
        print("\nInitializing model...")
        model = ChatOpenAI()

        # Test basic model functionality
        print("Testing basic model...")
        response = model.invoke([HumanMessage(content="hi!")])
        print(f"Basic response: {response.content}")

        # Bind tools to model
        model_with_tools = model.bind_tools(tools)

        # Test model without tool requirement
        print("\nTesting model with tools (no tool needed)...")
        response = model_with_tools.invoke([HumanMessage(content="Hi!")])
        print(f"Content: {response.content}")
        print(f"Tool calls: {response.tool_calls}")

        # Test model with tool requirement
        print("\nTesting model with tools (tool needed)...")
        response = model_with_tools.invoke(
            [HumanMessage(content="What's the weather in noida?")]
        )
        print(f"Content: {response.content}")
        print(f"Tool calls: {response.tool_calls}")

        # Get prompt template
        print("\nSetting up agent...")
        prompt = hub.pull("hwchase17/openai-functions-agent")
        print(f"Prompt messages count: {len(prompt.messages)}")

        # Create agent
        agent = create_tool_calling_agent(model, tools, prompt)

        # Create agent executor
        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,  # Added for better debugging
            handle_parsing_errors=True,  # Added for error handling
        )

        # Test agent interactions
        print("\n" + "=" * 50)
        print("TESTING AGENT INTERACTIONS")
        print("=" * 50)

        # Basic greeting
        print("\n1. Basic greeting:")
        result = agent_executor.invoke({"input": "hi!"})
        print(f"Response: {result['output']}")

        # LangSmith question (should use retriever tool)
        print("\n2. LangSmith question:")
        result = agent_executor.invoke({"input": "which language support mongo db?"})
        print(f"Response: {result['output']}")

        # Weather question (should use search tool)
        print("\n3. Weather question:")
        result = agent_executor.invoke({"input": "whats the weather in noida?"})
        print(f"Response: {result['output']}")

        # Chat with memory - introduction
        print("\n4. Introduction with empty chat history:")
        result = agent_executor.invoke(
            {"input": "hi! my name is naveen", "chat_history": []}
        )
        print(f"Response: {result['output']}")

        # Chat with memory - follow-up
        print("\n5. Follow-up with chat history:")
        result = agent_executor.invoke(
            {
                "chat_history": [
                    HumanMessage(content="hi! my name is naveen"),
                    AIMessage(content="Hello Naveen! How can I assist you today?"),
                ],
                "input": "what's my name?",
            }
        )
        print(f"Response: {result['output']}")

    except Exception as e:
        print(f"Error occurred: {e}")
        print("Please check your environment variables and API keys.")


if __name__ == "__main__":
    main()
