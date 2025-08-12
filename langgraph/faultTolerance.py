from typing import TypedDict, NotRequired
import time

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver

# -----------------------------
# Config
# -----------------------------
THREAD_ID = "thread-1"  # <- change if you want independent runs
SLEEP_SECONDS = 10  # <- how long step_2 "hangs" (interrupt here)


# -----------------------------
# State definition
# -----------------------------
class CrashState(TypedDict, total=False):
    """State carried across nodes."""

    input: str
    step1: NotRequired[str]
    step2: NotRequired[str]
    done: NotRequired[bool]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


# -----------------------------
# Nodes
# -----------------------------
def step_1(state: CrashState) -> CrashState:
    log("✅ step_1 executed")
    # Return only the fields you modify; LangGraph merges them into the state.
    return {"step1": "done", "input": state["input"]}


def step_2(state: CrashState) -> CrashState:
    log("⏳ step_2: simulating a long-running / stuck task")
    log(f"    (sleeping {SLEEP_SECONDS}s — press the STOP button to simulate a crash)")
    time.sleep(SLEEP_SECONDS)  # <-- interrupt here to simulate a crash
    return {"step2": "done"}


def step_3(state: CrashState) -> CrashState:
    log("✅ step_3 executed")
    return {"done": True}


# -----------------------------
# Build graph
# -----------------------------
builder = StateGraph(CrashState)
builder.add_node("step_1", step_1)
builder.add_node("step_2", step_2)
builder.add_node("step_3", step_3)

builder.set_entry_point("step_1")
builder.add_edge("step_1", "step_2")
builder.add_edge("step_2", "step_3")
builder.add_edge("step_3", END)

checkpointer = InMemorySaver()
graph = builder.compile(checkpointer=checkpointer)

# -----------------------------
# 1) First run — interrupt while step_2 is sleeping
# -----------------------------
try:
    log("▶️ Running graph (interrupt during step_2)...")
    graph.invoke(
        {"input": "start"},
        config={"configurable": {"thread_id": THREAD_ID}},
    )
except KeyboardInterrupt:
    log("❌ Manually interrupted — simulating a crash during step_2")

# -----------------------------
# 2) Resume — pass None + same thread_id to continue from last checkpoint
# -----------------------------
log("\n🔁 Resuming the graph to demonstrate fault tolerance...")
final_state = graph.invoke(
    None,  # <- resume from last saved state for this thread
    config={"configurable": {"thread_id": THREAD_ID}},
)
log(f"\n✅ Final state: {final_state}")

# -----------------------------
# (Optional) Inspect history + stream example
# -----------------------------
log("\n🧾 State history for this thread:")
history = list(graph.get_state_history({"configurable": {"thread_id": THREAD_ID}}))
for i, snapshot in enumerate(history, 1):
    # snapshot.values = dict of state keys -> values
    # snapshot.config["last_node"] (or similar) stores node name in newer versions
    last_node = snapshot.config.get("last_node", "unknown")
    keys = list(snapshot.values.keys())
    log(f"  {i}. last_node={last_node}  keys={keys}")

log("\n📡 Running again with .stream() to see live events (no interrupt):")
for event in graph.stream(
    {"input": "start"},
    config={"configurable": {"thread_id": f"{THREAD_ID}-stream-demo"}},
):
    # event is a dict like {'step_1': {...}}, {'step_2': {...}}, ...
    print(event)
