from typing import TypedDict, NotRequired
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI
import time

# -----------------------------
# Setup
# -----------------------------
load_dotenv()  # expects OPENAI_API_KEY in your environment

# Deterministic LLM for repeatable runs
llm = ChatOpenAI(temperature=0)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# -----------------------------
# State (partial updates; LangGraph merges for you)
# -----------------------------
class JokeState(TypedDict, total=False):
    topic: str
    joke: NotRequired[str]
    explanation: NotRequired[str]


# -----------------------------
# Nodes
# -----------------------------
def generate_joke(state: JokeState) -> JokeState:
    topic = state["topic"]
    prompt = f"Write one short, clean joke about: {topic}"
    response = llm.invoke(prompt).content
    log(f"✅ joke generated for topic='{topic}'")
    return {"joke": response}


def generate_explanation(state: JokeState) -> JokeState:
    joke = state["joke"]
    prompt = f"Explain why this joke is funny in 1–2 sentences:\n\n{joke}"
    response = llm.invoke(prompt).content
    log("✅ explanation generated")
    return {"explanation": response}


# -----------------------------
# Build the graph
# -----------------------------
graph = StateGraph(JokeState)
graph.add_node("generate_joke", generate_joke)
graph.add_node("generate_explanation", generate_explanation)

graph.add_edge(START, "generate_joke")
graph.add_edge("generate_joke", "generate_explanation")
graph.add_edge("generate_explanation", END)

checkpointer = InMemorySaver()  # use SqliteSaver/PG for real persistence
workflow = graph.compile(checkpointer=checkpointer)


# -----------------------------
# Helpers: history printing & safe snapshot selection
# -----------------------------
def print_history(title: str, hist):
    log(f"\n🧾 {title} (most recent last):")
    for i, snap in enumerate(hist, 1):
        # What you can safely rely on:
        # - snap.values      -> the merged state snapshot
        # - snap.config      -> config used to produce that snapshot (contains thread_id and checkpoint info)
        keys = list(snap.values.keys())
        # Try to surface a useful node hint if available (not guaranteed across versions)
        last_node = snap.config.get("last_node", "unknown")
        log(f"  {i}. last_node={last_node}  keys={keys}")


def latest_snapshot(hist):
    return hist[-1] if hist else None


def nth_from_end(hist, n=1):
    """n=1 => last; n=2 => second last, etc. Returns None if not enough history."""
    return hist[-n] if len(hist) >= n else None


# -----------------------------
# 1) Run two independent threads (isolation)
# -----------------------------
config1 = {"configurable": {"thread_id": "1"}}
config2 = {"configurable": {"thread_id": "2"}}

log("▶️ Running thread 1 with topic='pizza'")
workflow.invoke({"topic": "pizza"}, config=config1)

log("▶️ Running thread 2 with topic='pasta'")
workflow.invoke({"topic": "pasta"}, config=config2)

# Inspect latest state for thread 1
snap1 = workflow.get_state(config1)
log(f"\n📍 Thread 1 latest state keys: {list(snap1.values.keys())}")

# Full history per thread
hist1 = list(workflow.get_state_history(config1))
print_history("Thread 1 history", hist1)

# -----------------------------
# 2) Time travel in thread 1 (no hard-coded checkpoint IDs)
#    Pick an earlier snapshot and resume from it
# -----------------------------
# Choose a snapshot to "travel" to. For demo, take the second-to-last
travel_target = nth_from_end(hist1, n=2)
if travel_target:
    log("\n🕰️ Time-travel: resuming from a previous snapshot in thread 1")
    # The trick: reuse the snapshot's own config (contains thread_id + checkpoint info)
    travel_config = travel_target.config

    # a) Read the state *at* that snapshot
    snap_at_target = workflow.get_state(travel_config)
    log(f"  State at travel snapshot keys: {list(snap_at_target.values.keys())}")

    # b) Resume the workflow *from* that snapshot
    resumed = workflow.invoke(None, config=travel_config)
    log(f"  ✅ Resumed; latest keys now: {list(resumed.keys())}")
else:
    log("No suitable snapshot to time-travel to.")

# -----------------------------
# 3) Update state at a specific snapshot (edit & resume)
# -----------------------------
# We'll update the topic at the last snapshot of thread 1 and then resume
hist1 = list(workflow.get_state_history(config1))
target_for_edit = latest_snapshot(hist1)

if target_for_edit:
    edit_config = target_for_edit.config
    log("\n✏️ Updating topic at the chosen snapshot in thread 1 -> 'samosa'")
    workflow.update_state(edit_config, {"topic": "samosa"})

    # After editing state, run from that point again
    log("▶️ Resuming after update_state")
    workflow.invoke(None, config=edit_config)

    # Show history again to verify an additional checkpoint is created
    hist1_new = list(workflow.get_state_history(config1))
    print_history("Thread 1 history after update_state + resume", hist1_new)
else:
    log("No snapshot available to update in thread 1.")
