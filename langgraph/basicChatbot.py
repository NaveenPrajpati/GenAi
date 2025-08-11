from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langgraph.checkpoint.memory import MemorySaver, InMemorySaver

load_dotenv()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


graph = StateGraph(ChatState)


llm = ChatOpenAI()


def chatNode(state: ChatState):
    messages = state["messages"]
    response = llm.invoke(messages)

    return {"messages": [response]}


graph.add_node("chatNode", chatNode)
graph.add_edge(START, "chatNode")
graph.add_edge("chatNode", END)

memory = MemorySaver()

chatbot = graph.compile(checkpointer=memory)

test1 = {"messages": [HumanMessage(content="what is capital of bihar in india")]}

# result = chatbot.invoke(test1)
# print(result)
# print(result["messages"][-1].content)

thread_id = "1"
while True:
    userInput = input("Type question here: ")

    print("User: ", userInput)
    if userInput.strip().lower() in ["exit", "quit", "bye"]:
        break

    config = {"configurable": {"thread_id": thread_id}}
    response = chatbot.invoke(
        {"messages": [HumanMessage(content=userInput)]}, config=config
    )

    print("AI: ", response["messages"][-1].content)
