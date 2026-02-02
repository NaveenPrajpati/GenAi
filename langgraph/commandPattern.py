"""
Command Pattern - Dynamic Workflow Routing
============================================

LEARNING OBJECTIVES:
- Use Command for dynamic node navigation
- Build edge-less agent architectures
- Implement flexible control flow
- Combine state updates with routing

CONCEPT:
The Command pattern (new in LangGraph 1.0+) allows nodes to dynamically
decide which node to execute next AND update state in a single return.
This enables more flexible architectures without pre-defined edges.

TRADITIONAL vs COMMAND:
    ┌─────────────────────────────────────────────────────────────┐
    │               Traditional vs Command Pattern                 │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  TRADITIONAL (Static Edges):                                │
    │  - Edges defined at compile time                            │
    │  - Conditional edges for branching                          │
    │  - Routing logic separate from nodes                        │
    │                                                              │
    │  COMMAND (Dynamic):                                         │
    │  - Nodes decide next step at runtime                        │
    │  - No need for pre-defined edges                            │
    │  - State update + routing in one return                     │
    │  - More flexible, self-contained nodes                      │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

COMMAND STRUCTURE:
    Command(
        goto="next_node",      # Which node to execute next
        update={"key": value}, # State updates to apply
        resume=value           # For resuming from interrupt
    )

VISUAL COMPARISON:
    Traditional:                    Command Pattern:
    ┌─────────┐                    ┌─────────┐
    │  Node   │──edge──┐           │  Node   │
    └─────────┘        │           └────┬────┘
         ↓             ▼                │
    (routing func)  ┌─────┐             │ Command(goto="B")
         ↓          │  B  │             ▼
    ┌─────────┐     └─────┘        ┌─────────┐
    │    A    │                    │    B    │
    └─────────┘                    └─────────┘

PREREQUISITES:
- Completed: langgraph/conditionalWorkflow.py
- Understanding of StateGraph
- OpenAI API key in .env

NEXT STEPS:
- practicsProjects/6multiAgent.py - Multi-agent systems
- langgraph/humanInTheLoop.py - Combine with interrupts
"""

from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command, interrupt
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

load_dotenv()

# =============================================================================
# PART 1: Basic Command Pattern
# =============================================================================


class TaskState(TypedDict):
    """State for task processing."""
    task: str
    category: str
    priority: str
    result: str
    steps_taken: list[str]


def classify_task(state: TaskState) -> Command[Literal["urgent_handler", "normal_handler", "low_priority"]]:
    """Classify task and dynamically route to appropriate handler."""
    task = state["task"].lower()
    steps = state.get("steps_taken", []) + ["classify_task"]

    # Determine category and priority
    if "urgent" in task or "emergency" in task or "critical" in task:
        return Command(
            goto="urgent_handler",
            update={
                "category": "critical",
                "priority": "P0",
                "steps_taken": steps
            }
        )
    elif "bug" in task or "error" in task or "fix" in task:
        return Command(
            goto="normal_handler",
            update={
                "category": "bug",
                "priority": "P1",
                "steps_taken": steps
            }
        )
    else:
        return Command(
            goto="low_priority",
            update={
                "category": "enhancement",
                "priority": "P2",
                "steps_taken": steps
            }
        )


def urgent_handler(state: TaskState) -> Command[Literal["__end__"]]:
    """Handle urgent tasks immediately."""
    steps = state.get("steps_taken", []) + ["urgent_handler"]

    return Command(
        goto=END,
        update={
            "result": f"URGENT: Task '{state['task']}' escalated to on-call team!",
            "steps_taken": steps
        }
    )


def normal_handler(state: TaskState) -> Command[Literal["review"]]:
    """Handle normal priority tasks."""
    steps = state.get("steps_taken", []) + ["normal_handler"]

    return Command(
        goto="review",
        update={
            "result": f"Processing bug report: {state['task']}",
            "steps_taken": steps
        }
    )


def low_priority_handler(state: TaskState) -> Command[Literal["__end__"]]:
    """Handle low priority tasks."""
    steps = state.get("steps_taken", []) + ["low_priority_handler"]

    return Command(
        goto=END,
        update={
            "result": f"Added to backlog: {state['task']}",
            "steps_taken": steps
        }
    )


def review_task(state: TaskState) -> Command[Literal["__end__"]]:
    """Review processed tasks."""
    steps = state.get("steps_taken", []) + ["review"]

    return Command(
        goto=END,
        update={
            "result": state["result"] + " | Reviewed and assigned.",
            "steps_taken": steps
        }
    )


# Build graph with Command pattern - minimal edges needed
task_graph = StateGraph(TaskState)
task_graph.add_node("classify", classify_task)
task_graph.add_node("urgent_handler", urgent_handler)
task_graph.add_node("normal_handler", normal_handler)
task_graph.add_node("low_priority", low_priority_handler)
task_graph.add_node("review", review_task)

# Only need the entry edge - Command handles the rest!
task_graph.add_edge(START, "classify")

task_app = task_graph.compile()


def demo_basic_command():
    """Demonstrate basic Command pattern."""
    print("=" * 70)
    print("DEMO 1: Basic Command Pattern")
    print("=" * 70)

    test_tasks = [
        "URGENT: Server down in production!",
        "Bug: Login button not working",
        "Add dark mode to settings page"
    ]

    for task in test_tasks:
        print(f"\n--- Task: {task[:40]}... ---")
        result = task_app.invoke({
            "task": task,
            "steps_taken": []
        })
        print(f"Category: {result['category']}")
        print(f"Priority: {result['priority']}")
        print(f"Result: {result['result']}")
        print(f"Path: {' -> '.join(result['steps_taken'])}")


# =============================================================================
# PART 2: Multi-Agent Routing with Command
# =============================================================================


class AgentState(TypedDict):
    """State for multi-agent system."""
    messages: Annotated[list[BaseMessage], add_messages]
    current_agent: str
    task_complete: bool


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def supervisor(state: AgentState) -> Command[Literal["researcher", "writer", "reviewer", "__end__"]]:
    """Supervisor decides which agent should work next."""
    messages = state["messages"]
    last_message = messages[-1].content if messages else ""

    print(f"  [Supervisor] Analyzing: {last_message[:50]}...")

    # Determine which agent should handle this
    if state.get("task_complete"):
        return Command(goto=END)

    if not any(msg for msg in messages if isinstance(msg, AIMessage) and "research" in msg.content.lower()):
        return Command(
            goto="researcher",
            update={"current_agent": "researcher"}
        )
    elif not any(msg for msg in messages if isinstance(msg, AIMessage) and "draft" in msg.content.lower()):
        return Command(
            goto="writer",
            update={"current_agent": "writer"}
        )
    else:
        return Command(
            goto="reviewer",
            update={"current_agent": "reviewer"}
        )


def researcher(state: AgentState) -> Command[Literal["supervisor"]]:
    """Research agent gathers information."""
    print("  [Researcher] Gathering information...")

    # Simulate research
    research_result = AIMessage(
        content="Research complete: Found 3 key sources about the topic. "
                "Main findings include historical context and current trends."
    )

    return Command(
        goto="supervisor",
        update={"messages": [research_result]}
    )


def writer(state: AgentState) -> Command[Literal["supervisor"]]:
    """Writer agent creates content."""
    print("  [Writer] Creating draft...")

    # Simulate writing
    draft = AIMessage(
        content="Draft complete: Based on the research, here's a comprehensive "
                "overview of the topic with clear explanations and examples."
    )

    return Command(
        goto="supervisor",
        update={"messages": [draft]}
    )


def reviewer(state: AgentState) -> Command[Literal["supervisor"]]:
    """Reviewer agent checks quality."""
    print("  [Reviewer] Reviewing content...")

    # Simulate review
    review = AIMessage(
        content="Review complete: Content is accurate and well-structured. "
                "Approved for publication."
    )

    return Command(
        goto="supervisor",
        update={
            "messages": [review],
            "task_complete": True
        }
    )


# Build multi-agent graph
agent_graph = StateGraph(AgentState)
agent_graph.add_node("supervisor", supervisor)
agent_graph.add_node("researcher", researcher)
agent_graph.add_node("writer", writer)
agent_graph.add_node("reviewer", reviewer)

agent_graph.add_edge(START, "supervisor")

agent_app = agent_graph.compile()


def demo_multi_agent_command():
    """Demonstrate multi-agent routing with Command."""
    print("\n" + "=" * 70)
    print("DEMO 2: Multi-Agent Routing with Command")
    print("=" * 70)

    result = agent_app.invoke({
        "messages": [HumanMessage(content="Write an article about AI trends")],
        "current_agent": "supervisor",
        "task_complete": False
    })

    print("\n--- Final Messages ---")
    for msg in result["messages"]:
        role = "User" if isinstance(msg, HumanMessage) else "Agent"
        print(f"{role}: {msg.content[:80]}...")


# =============================================================================
# PART 3: Command with Interrupt (Combined Pattern)
# =============================================================================


class ApprovalState(TypedDict):
    """State for approval workflow with Command."""
    action: str
    risk_score: int
    approved: bool
    status: str


def assess_action(state: ApprovalState) -> Command[Literal["approve", "execute", "__end__"]]:
    """Assess action and route based on risk."""
    action = state["action"]

    # Calculate risk score
    risk = 0
    if "delete" in action.lower():
        risk += 5
    if "all" in action.lower():
        risk += 3
    if "production" in action.lower():
        risk += 5

    print(f"  [Assess] Action: {action}, Risk Score: {risk}")

    if risk >= 5:
        return Command(
            goto="approve",
            update={"risk_score": risk}
        )
    else:
        return Command(
            goto="execute",
            update={"risk_score": risk, "approved": True}
        )


def request_approval(state: ApprovalState) -> Command[Literal["execute", "__end__"]]:
    """Request human approval using interrupt."""
    print(f"  [Approve] High risk action detected (score: {state['risk_score']})")

    # Use interrupt for human approval
    response = interrupt(
        f"HIGH RISK ACTION: {state['action']}\n"
        f"Risk Score: {state['risk_score']}\n"
        "Type 'approve' to continue or 'reject' to cancel."
    )

    if response.lower() == "approve":
        return Command(
            goto="execute",
            update={"approved": True}
        )
    else:
        return Command(
            goto=END,
            update={
                "approved": False,
                "status": "Action rejected by human reviewer."
            }
        )


def execute_action(state: ApprovalState) -> Command[Literal["__end__"]]:
    """Execute the approved action."""
    print(f"  [Execute] Executing: {state['action']}")

    return Command(
        goto=END,
        update={
            "status": f"Successfully executed: {state['action']}"
        }
    )


# Build approval graph with Command + interrupt
approval_graph = StateGraph(ApprovalState)
approval_graph.add_node("assess", assess_action)
approval_graph.add_node("approve", request_approval)
approval_graph.add_node("execute", execute_action)

approval_graph.add_edge(START, "assess")

memory = InMemorySaver()
approval_app = approval_graph.compile(checkpointer=memory)


def demo_command_with_interrupt():
    """Demonstrate Command combined with interrupt."""
    print("\n" + "=" * 70)
    print("DEMO 3: Command + Interrupt Pattern")
    print("=" * 70)

    # Test 1: Low risk (no interrupt)
    print("\n--- Test 1: Low Risk Action ---")
    result1 = approval_app.invoke({
        "action": "Update user preferences",
        "approved": False
    }, config={"configurable": {"thread_id": "test1"}})
    print(f"Status: {result1.get('status', 'In progress')}")

    # Test 2: High risk (triggers interrupt)
    print("\n--- Test 2: High Risk Action ---")
    config2 = {"configurable": {"thread_id": "test2"}}

    result2 = approval_app.invoke({
        "action": "Delete all production data",
        "approved": False
    }, config=config2)

    current_state = approval_app.get_state(config2)
    if current_state.next:
        print(">>> Interrupt triggered - awaiting approval <<<")
        user_input = input("Enter decision (approve/reject): ")

        final_result = approval_app.invoke(
            Command(resume=user_input),
            config=config2
        )
        print(f"Final Status: {final_result['status']}")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("COMMAND PATTERN DEMOS")
    print("=" * 70)

    demo_basic_command()
    demo_multi_agent_command()
    demo_command_with_interrupt()

    print("\n" + "=" * 70)
    print("COMMAND PATTERN BEST PRACTICES")
    print("=" * 70)
    print("""
    1. Use Command for dynamic routing within nodes
    2. Combine goto and update for atomic state+routing changes
    3. Use Literal types for type safety in goto destinations
    4. Command(goto=END) to terminate the workflow
    5. Combine with interrupt() for human-in-the-loop
    6. Use resume parameter to continue from interrupts
    7. Minimal edges needed - just START to entry node
    8. Each node is self-contained with its routing logic
    9. Great for supervisor/worker multi-agent patterns
    10. More flexible than conditional_edges for complex logic
    """)
