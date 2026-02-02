"""
Human-in-the-Loop (HIL) - Interrupt and Resume Workflows
==========================================================

LEARNING OBJECTIVES:
- Use interrupt() for dynamic workflow pauses
- Get human approval before critical actions
- Resume workflows from interruption points
- Implement approval workflows

CONCEPT:
Human-in-the-Loop allows workflows to pause execution, request human
input or approval, and then resume from exactly where they left off.
This is critical for:
- Approving sensitive actions
- Reviewing AI decisions
- Collecting additional input
- Audit and compliance

TWO APPROACHES:
    ┌─────────────────────────────────────────────────────────────┐
    │                  Human-in-the-Loop Methods                   │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  1. Static Breakpoints (interrupt_before/after):            │
    │     - Defined at compile time                               │
    │     - Always pause at specific nodes                        │
    │     - Good for debugging and fixed approval points          │
    │                                                              │
    │  2. Dynamic Interrupts (interrupt()):                       │
    │     - Called within node execution                          │
    │     - Conditional pausing based on logic                    │
    │     - Returns a value when resumed                          │
    │     - More flexible for complex workflows                   │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

INTERRUPT FLOW:
    ┌──────┐     ┌──────────┐     ┌─────────────┐     ┌──────┐
    │ Node │ --> │ interrupt│ --> │ Human Input │ --> │Resume│
    │  A   │     │   ()     │     │   Required  │     │      │
    └──────┘     └──────────┘     └─────────────┘     └──────┘
                      ↓                                   ↓
                 Workflow                            Continue
                  Paused                             Execution

PREREQUISITES:
- Completed: langgraph/persistance.py
- Understanding of checkpointing
- OpenAI API key in .env

NEXT STEPS:
- langgraph/commandPattern.py - Dynamic routing
- practicsProjects/1tool-use.py - Production patterns
"""

from typing import Annotated, TypedDict, Literal
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage, SystemMessage

load_dotenv()

# =============================================================================
# PART 1: Static Breakpoints (interrupt_before/after)
# =============================================================================


class ApprovalState(TypedDict):
    """State for approval workflow."""
    request: str
    analysis: str
    approved: bool
    result: str


def analyze_request(state: ApprovalState) -> ApprovalState:
    """Analyze the incoming request."""
    print("  [analyze] Analyzing request...")
    return {
        "analysis": f"Analysis of '{state['request']}': This action will modify the database."
    }


def execute_action(state: ApprovalState) -> ApprovalState:
    """Execute the approved action."""
    if state.get("approved"):
        print("  [execute] Executing approved action...")
        return {"result": f"Successfully executed: {state['request']}"}
    else:
        print("  [execute] Action was not approved.")
        return {"result": "Action was rejected by human reviewer."}


# Build graph with static breakpoint
static_graph = StateGraph(ApprovalState)
static_graph.add_node("analyze", analyze_request)
static_graph.add_node("execute", execute_action)

static_graph.add_edge(START, "analyze")
static_graph.add_edge("analyze", "execute")
static_graph.add_edge("execute", END)

# Compile with interrupt_before - always pause before 'execute' node
memory1 = InMemorySaver()
static_app = static_graph.compile(
    checkpointer=memory1,
    interrupt_before=["execute"]  # Pause BEFORE this node
)


def demo_static_breakpoint():
    """Demonstrate static breakpoint pattern."""
    print("=" * 70)
    print("DEMO 1: Static Breakpoint (interrupt_before)")
    print("=" * 70)

    config = {"configurable": {"thread_id": "demo-1"}}
    initial_state = {"request": "Delete all user records", "approved": False}

    # Run until breakpoint
    print("\nPhase 1: Run until breakpoint")
    print("-" * 40)
    result = static_app.invoke(initial_state, config=config)
    print(f"Analysis: {result.get('analysis')}")
    print(">>> Workflow paused at 'execute' node <<<")

    # Simulate human review
    print("\nPhase 2: Human Review")
    print("-" * 40)
    print("Human reviewer sees: ", result.get("analysis"))
    human_decision = input("Approve this action? (yes/no): ").lower().strip()

    # Update state with human decision
    static_app.update_state(
        config,
        {"approved": human_decision == "yes"}
    )

    # Resume execution
    print("\nPhase 3: Resume Execution")
    print("-" * 40)
    final_result = static_app.invoke(None, config=config)
    print(f"Final Result: {final_result.get('result')}")


# =============================================================================
# PART 2: Dynamic Interrupts (interrupt() function)
# =============================================================================


class TransferState(TypedDict):
    """State for money transfer workflow."""
    from_account: str
    to_account: str
    amount: float
    risk_level: str
    human_approval: str
    status: str


def assess_risk(state: TransferState) -> TransferState:
    """Assess the risk level of the transfer."""
    print("  [assess_risk] Evaluating transfer...")

    amount = state["amount"]
    if amount > 10000:
        risk = "high"
    elif amount > 1000:
        risk = "medium"
    else:
        risk = "low"

    return {"risk_level": risk}


def request_approval(state: TransferState) -> TransferState:
    """Request human approval for high-risk transfers."""
    print("  [request_approval] Checking if approval needed...")

    if state["risk_level"] == "high":
        # Dynamic interrupt - only triggers for high-risk transfers
        print("  >>> HIGH RISK DETECTED - Requesting human approval <<<")

        # interrupt() pauses execution and returns the value provided when resumed
        approval = interrupt(
            f"APPROVAL REQUIRED: Transfer ${state['amount']} from "
            f"{state['from_account']} to {state['to_account']}. Approve? (yes/no)"
        )

        return {"human_approval": approval}
    else:
        # Low/medium risk - auto-approve
        return {"human_approval": "auto-approved"}


def process_transfer(state: TransferState) -> TransferState:
    """Process the transfer based on approval status."""
    print("  [process_transfer] Processing...")

    approval = state.get("human_approval", "")

    if approval in ["yes", "auto-approved"]:
        return {
            "status": f"COMPLETED: Transferred ${state['amount']} "
                      f"from {state['from_account']} to {state['to_account']}"
        }
    else:
        return {
            "status": f"REJECTED: Transfer of ${state['amount']} was denied."
        }


# Build dynamic interrupt graph
dynamic_graph = StateGraph(TransferState)
dynamic_graph.add_node("assess_risk", assess_risk)
dynamic_graph.add_node("request_approval", request_approval)
dynamic_graph.add_node("process_transfer", process_transfer)

dynamic_graph.add_edge(START, "assess_risk")
dynamic_graph.add_edge("assess_risk", "request_approval")
dynamic_graph.add_edge("request_approval", "process_transfer")
dynamic_graph.add_edge("process_transfer", END)

memory2 = InMemorySaver()
dynamic_app = dynamic_graph.compile(checkpointer=memory2)


def demo_dynamic_interrupt():
    """Demonstrate dynamic interrupt pattern."""
    print("\n" + "=" * 70)
    print("DEMO 2: Dynamic Interrupt (interrupt() function)")
    print("=" * 70)

    # Test 1: Low-risk transfer (no interrupt)
    print("\n--- Test 1: Low-Risk Transfer ($500) ---")
    config1 = {"configurable": {"thread_id": "transfer-1"}}
    result1 = dynamic_app.invoke({
        "from_account": "checking",
        "to_account": "savings",
        "amount": 500
    }, config=config1)
    print(f"Result: {result1['status']}")

    # Test 2: High-risk transfer (triggers interrupt)
    print("\n--- Test 2: High-Risk Transfer ($15,000) ---")
    config2 = {"configurable": {"thread_id": "transfer-2"}}

    # Phase 1: Run until interrupt
    print("\nPhase 1: Run until interrupt")
    result2 = dynamic_app.invoke({
        "from_account": "checking",
        "to_account": "external",
        "amount": 15000
    }, config=config2)

    # Check if we're in an interrupted state
    current_state = dynamic_app.get_state(config2)
    if current_state.next:  # If there are pending nodes, we're interrupted
        print(f"\n>>> Workflow interrupted <<<")
        print(f"Pending node: {current_state.next}")

        # Get the interrupt message
        # The interrupt value is in the state's interrupts
        print("\nPhase 2: Human provides approval")
        human_input = input("Enter your decision (yes/no): ").lower().strip()

        # Resume with human input using Command
        print("\nPhase 3: Resume with human input")
        final_result = dynamic_app.invoke(
            Command(resume=human_input),
            config=config2
        )
        print(f"Final Result: {final_result['status']}")
    else:
        print(f"Result: {result2['status']}")


# =============================================================================
# PART 3: Tool Approval Pattern
# =============================================================================


class ToolState(TypedDict):
    """State for tool approval workflow."""
    messages: Annotated[list[BaseMessage], add_messages]
    pending_tool: dict | None
    tool_result: str | None


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def plan_action(state: ToolState) -> ToolState:
    """Plan which tool to use."""
    last_message = state["messages"][-1].content

    # Simulate tool selection
    if "delete" in last_message.lower():
        return {
            "pending_tool": {
                "name": "delete_file",
                "args": {"filename": "important.txt"}
            }
        }
    elif "send" in last_message.lower():
        return {
            "pending_tool": {
                "name": "send_email",
                "args": {"to": "user@example.com", "subject": "Test"}
            }
        }
    else:
        return {"pending_tool": None}


def get_approval(state: ToolState) -> ToolState:
    """Get human approval for dangerous tools."""
    tool = state.get("pending_tool")

    if tool and tool["name"] in ["delete_file", "send_email"]:
        # Request approval for dangerous operations
        approval = interrupt(
            f"Tool '{tool['name']}' wants to execute with args: {tool['args']}\n"
            "Reply 'yes' to approve or 'no' to reject."
        )

        if approval.lower() != "yes":
            return {
                "tool_result": f"Tool {tool['name']} was rejected by user.",
                "pending_tool": None
            }

    return {}


def execute_tool(state: ToolState) -> ToolState:
    """Execute the approved tool."""
    tool = state.get("pending_tool")

    if tool:
        # Simulate tool execution
        result = f"Executed {tool['name']} with {tool['args']}"
        return {
            "tool_result": result,
            "messages": [AIMessage(content=result)]
        }
    elif state.get("tool_result"):
        return {"messages": [AIMessage(content=state["tool_result"])]}
    else:
        return {"messages": [AIMessage(content="No action needed.")]}


# Build tool approval graph
tool_graph = StateGraph(ToolState)
tool_graph.add_node("plan", plan_action)
tool_graph.add_node("approve", get_approval)
tool_graph.add_node("execute", execute_tool)

tool_graph.add_edge(START, "plan")
tool_graph.add_edge("plan", "approve")
tool_graph.add_edge("approve", "execute")
tool_graph.add_edge("execute", END)

memory3 = InMemorySaver()
tool_app = tool_graph.compile(checkpointer=memory3)


def demo_tool_approval():
    """Demonstrate tool approval pattern."""
    print("\n" + "=" * 70)
    print("DEMO 3: Tool Approval Pattern")
    print("=" * 70)

    config = {"configurable": {"thread_id": "tool-demo"}}

    # Request that will trigger approval
    print("\nUser request: 'Please delete the temp files'")
    result = tool_app.invoke({
        "messages": [HumanMessage(content="Please delete the temp files")]
    }, config=config)

    # Check if interrupted
    current_state = tool_app.get_state(config)
    if current_state.next:
        print(f"\n>>> Approval required for tool execution <<<")

        approval = input("Approve tool execution? (yes/no): ").lower().strip()

        final_result = tool_app.invoke(
            Command(resume=approval),
            config=config
        )
        print(f"\nFinal Result: {final_result['messages'][-1].content}")
    else:
        print(f"Result: {result['messages'][-1].content}")


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    print("HUMAN-IN-THE-LOOP DEMOS")
    print("=" * 70)

    # Run demos
    demo_static_breakpoint()
    demo_dynamic_interrupt()
    demo_tool_approval()

    # Best practices
    print("\n" + "=" * 70)
    print("HUMAN-IN-THE-LOOP BEST PRACTICES")
    print("=" * 70)
    print("""
    1. Use interrupt_before/after for fixed checkpoints (debugging, compliance)
    2. Use interrupt() for conditional, dynamic pauses
    3. Always use a checkpointer to preserve state during interrupts
    4. Include clear context in interrupt messages
    5. Handle rejection cases gracefully
    6. Consider timeout for human responses in production
    7. Log all approval decisions for audit trails
    8. Use Command(resume=value) to continue after interrupt()
    9. Check state.next to determine if workflow is interrupted
    10. Combine with streaming for responsive UIs
    """)
