from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated, List
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage


class TeamState(TypedDict, total=False):
    task: str
    plan: list[str]
    sources: list[dict]
    analysis: dict
    draft: str
    citations: list[dict]
    errors: list[str]


llm = init_chat_model(model="openai:gpt-4o-mini")


def supervisor(s: TeamState):
    if not s.get("plan"):
        # Make a plan first
        plan = llm.invoke(
            [
                HumanMessage(
                    content=f"Task: {s['task']}. Break into 3 steps: research, analysis, writing."
                )
            ]
        ).content
        return {"plan": [plan]}
    # Check halt condition
    if len(s.get("sources", [])) < 2:
        return {"next": "halt"}
    elif not s.get("analysis"):
        return {"next": "analyst"}
    elif not s.get("draft"):
        return {"next": "writer"}
    else:
        return {}


def researcher(s: TeamState):
    # search/fetch sources (stub)
    return {
        "sources": [
            {
                "url": "https://example.com",
                "text": "Tool A costs $10",
                "credibility_score": 0.9,
            }
        ]
    }


def analyst(s: TeamState):
    # parse sources into table
    return {"analysis": {"pricing_table": [("Tool A", "$10")]}}


def writer(s: TeamState):
    draft = f"""Brief on {s['task']}
Pricing: {s['analysis']['pricing_table']}
Sources: {[src['url'] for src in s['sources']]}"""
    return {
        "draft": draft,
        "citations": [{"url": u} for u in [src["url"] for src in s["sources"]]],
    }


def halt(s: TeamState):
    return {"errors": ["Insufficient credible sources (<2)."], "draft": "Halted."}


g = StateGraph(TeamState)
g.add_node("supervisor", supervisor)
g.add_node("researcher", researcher)
g.add_node("analyst", analyst)
g.add_node("writer", writer)
g.add_node("halt", halt)

g.add_edge(START, "supervisor")
g.add_edge("supervisor", "researcher")
g.add_edge("researcher", "supervisor")
g.add_edge("supervisor", "analyst")
g.add_edge("analyst", "writer")
g.add_edge("writer", END)
g.add_edge("supervisor", "halt")
g.add_edge("halt", END)

graph = g.compile()
