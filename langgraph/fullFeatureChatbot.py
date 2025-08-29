# chatbot.py
from typing import Annotated
from typing_extensions import TypedDict

from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import ToolMessage

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from dotenv import load_dotenv

load_dotenv()


# ---------- State ----------
class State(TypedDict):
    messages: Annotated[list, add_messages]
    # optional custom fields (used in step 5)
    name: str
    birthday: str


# ---------- Graph ----------
graph_builder = StateGraph(State)

# Choose a model (swap providers by changing this id)
llm = init_chat_model("openai:gpt-4.1")  # or "anthropic:claude-3-5-sonnet-latest", etc.

# Tools (step 2)
search = TavilySearch(max_results=2)


@tool
def human_assistance(
    name: str = "",
    birthday: str = "",
    tool_call_id: Annotated[str, InjectedToolCallId] = "",
) -> str:
    """Pause, ask a human to verify/correct (HITL)."""
    payload = {
        "question": "Is this correct?",
        "name": name,
        "birthday": birthday,
    }
    human = interrupt(payload)  # Pauses execution until resumed

    # If no custom fields were provided, just relay general guidance
    if not (name or birthday):
        return human["data"]

    # If fields were provided, write a state update + tool message
    correct = str(human.get("correct", "")).lower().startswith("y")
    new_name = name if correct else human.get("name", name)
    new_bday = birthday if correct else human.get("birthday", birthday)
    note = "Correct." if correct else f"Corrected: {human}"

    return Command(
        update={
            "name": new_name,
            "birthday": new_bday,
            "messages": [ToolMessage(note, tool_call_id=tool_call_id)],
        }
    )


tools = [search, human_assistance]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    # If you plan to interrupt during tool execution, you may want to keep
    # tool calls to one at a time (per HITL tutorial guidance).
    msg = llm_with_tools.invoke(state["messages"])
    # assert len(getattr(msg, "tool_calls", [])) <= 1
    return {"messages": [msg]}


graph_builder.add_node("chatbot", chatbot)
graph_builder.add_node("tools", ToolNode(tools=tools))
graph_builder.add_conditional_edges(
    "chatbot", tools_condition, {"tools": "tools", END: END}
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.set_entry_point("chatbot")

# Memory / persistence (step 3)
memory = InMemorySaver()
graph = graph_builder.compile(checkpointer=memory)


# ---------- Helpers ----------
def ask(user_text: str, thread_id: str = "demo"):
    config = {"configurable": {"thread_id": thread_id}}
    for ev in graph.stream(
        {"messages": [{"role": "user", "content": user_text}]},
        config,
        stream_mode="values",
    ):
        if "messages" in ev:
            print("Assistant:", ev["messages"][-1].content)


def resume_human(data, thread_id: str = "demo"):
    config = {"configurable": {"thread_id": thread_id}}
    for ev in graph.stream(Command(resume=data), config, stream_mode="values"):
        if "messages" in ev:
            print("Assistant:", ev["messages"][-1].content)


def time_travel_resume(predicate, thread_id: str = "demo"):
    config = {"configurable": {"thread_id": thread_id}}
    to_replay = next(s for s in graph.get_state_history(config) if predicate(s))
    for ev in graph.stream(None, to_replay.config, stream_mode="values"):
        if "messages" in ev:
            print("Assistant:", ev["messages"][-1].content)


# ---------- Demo ----------
if __name__ == "__main__":
    tid = "demo"

    print("\n[1] Basic + Tools + Memory")
    ask("Hi! I'm Naveen.", tid)
    ask("What did I just tell you about my name?", tid)
    ask("Find recent info about LangGraph.", tid)

    print("\n[2] Human-in-the-loop")
    ask("Please ask a human for agent-building tips.", tid)
    # (your app/console would capture the pause and provide a resume payload)
    resume_human({"data": "Use LangGraph with checkpoints and interrupts."}, tid)

    print("\n[3] Customize State (verify fields via human tool)")
    ask(
        "Look up LangGraph release date and send to human_assistance for verification.",
        tid,
    )
    resume_human(
        {"name": "LangGraph", "birthday": "Jan 17, 2024", "correct": "yes"}, tid
    )

    print("\n[4] Time travel (resume from the last point where tools will run next)")
    time_travel_resume(lambda s: "tools" in s.next, tid)
