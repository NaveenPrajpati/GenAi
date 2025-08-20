"""
The essential components
        1.	State (schema & channels)
A single source-of-truth object shared by the whole graph. You define keys (“channels”) and, for some of them, a reducer that tells LangGraph how to merge updates (e.g., append messages instead of overwrite).  ￼ ￼
        2.	Nodes (the work units)
Pure Python callables (LLM calls, tools, business logic). Each node reads part of the state and returns updates to state channels.  ￼
        3.	Edges (flow control)
Connect nodes in sequence or branch conditionally. You can route based on model output (e.g., “did the model ask to call a tool?”).  ￼
        4.	Compile step
Turns your builder into an executable graph and is where you plug in runtime options like checkpointers and static breakpoints.  ￼
        5.	Tools & ToolNode
Wrap external actions/APIs as LangChain tools and execute them with the prebuilt ToolNode; pair with a condition like tools_condition to route into tools only when the model requested them.  ￼
        6.	LLM(s) & prompts
Any chat model via init_chat_model(...); you can mark parts of the graph configurable so you can swap models/system prompts at run time.  ￼
        7.	Memory & persistence (checkpointers)
Add durable memory by compiling with a checkpointer (e.g., in-memory, SQLite, Postgres). Provide a thread_id when invoking to get long-term, cross-session memory—and unlock time travel.  ￼ ￼
        8.	Human-in-the-loop (HIL)
Use interrupt() for dynamic breakpoints to pause, get human input/approval, then resume exactly where you left off. Static interrupt_before/after are handy for debugging.  ￼ ￼
        9.	Retries & caching (resilience)
Per-node RetryPolicy and optional CachePolicy to handle flaky APIs or expensive steps.  ￼
        10.	Streaming & events
Stream state updates or full values after each step if you’re building chat UIs or dashboards.  ￼
        11.	Observability & tooling
Run locally or deploy with LangGraph Server/Platform; debug visually in LangGraph Studio and trace/evaluate with LangSmith.  ￼ ￼

"""

from typing import Annotated, TypedDict, List
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import ToolNode, tools_condition
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver


# 1) State schema: keep a growing chat transcript
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


# 2) LLM node: decide what to do next (answer or call a tool)
llm = init_chat_model(model="openai:gpt-4o-mini")  # pick your provider/model


@tool
def get_time() -> str:
    """Returns the current server time."""
    import datetime as dt

    return dt.datetime.now().isoformat(timespec="seconds")


tools = [get_time]
tool_node = ToolNode(tools)


def call_model(state: AgentState):
    # bind tools so the model can decide to call one
    msg = llm.bind_tools(tools).invoke(state["messages"])
    return {"messages": [msg]}


def respond(state: AgentState):
    # simple “final answer” step
    last = state["messages"][-1]
    if isinstance(last, HumanMessage):
        reply = llm.invoke([last])
    else:
        reply = AIMessage(content="Done.")
    return {"messages": [reply]}


# 3) Build graph: nodes + edges (with tool routing)
builder = StateGraph(AgentState)
builder.add_node("agent", call_model)
builder.add_node("tools", tool_node)
builder.add_node("respond", respond)

builder.add_edge(START, "agent")
# if model requested tool(s), go to tools; otherwise go to respond
builder.add_conditional_edges("agent", tools_condition)
builder.add_edge("tools", "agent")  # loop after tool results
builder.add_edge("respond", END)

# 4) Persistence (SQLite) -> long-term memory, time-travel, HIL
conn = sqlite3.connect("checkpoints.sqlite", check_same_thread=False)
memory = SqliteSaver(conn)
graph = builder.compile(checkpointer=memory)

# 5) Run with a thread id to enable memory across calls
thread_cfg = {"configurable": {"thread_id": "user-123"}}

result = graph.invoke(
    {"messages": [HumanMessage(content="What time is it?")]}, config=thread_cfg
)
print(result["messages"][-1].content)
