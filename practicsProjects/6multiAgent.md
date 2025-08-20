# Multi-Agent Workflow with LangGraph - Implementation Guide

## Project Goal
Build a Supervisor → (Researcher, Analyst, Writer) multi-agent system with shared memory & hand-off rules.

**Success Criteria**: Given a prompt like "Compare 3 newsletter tools for a solo founder", the team:
- Plans, collects sources, analyzes pricing, then produces a 1-page brief with a table + citations
- **Bonus**: Add a "halt" path if sources are too weak (<2 credible)

## Architecture Overview

- **Supervisor Agent**: Orchestrates the workflow, makes routing decisions, and maintains shared state
- **Worker Agents**: Researcher → Analyst → Writer (sequential with conditional branching)
- **Shared Memory**: LangGraph state object containing all intermediate results

## 1. State Schema Design

```python
from typing import TypedDict, List, Dict, Optional
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    messages: List[BaseMessage]
    task: str  # Original request
    research_plan: Optional[str]
    sources: List[Dict]  # URLs, credibility scores, content
    analysis_results: Dict  # Pricing, features, pros/cons
    final_brief: Optional[str]
    current_agent: str
    halt_reason: Optional[str]  # If workflow needs to stop
    iteration_count: int
```

## 2. Agent Definitions

### Supervisor Agent
- Routes to next agent based on current state
- Validates completion criteria
- Decides on halt conditions

### Researcher Agent
- Creates search strategy
- Collects sources (web search)
- Validates source credibility
- Triggers halt if <2 credible sources found

### Analyst Agent
- Processes collected sources
- Extracts pricing/feature data
- Creates comparison framework

### Writer Agent
- Synthesizes analysis into brief
- Formats table and citations
- Produces final deliverable

## 3. Graph Structure

```
START → Supervisor → Researcher → [Credibility Check]
                                      ↓
                              [Halt] ← → Analyst → Writer → END
                                                    ↓
                                              Supervisor (review)
```

## 4. Hand-off Rules

- **Supervisor → Researcher**: Always first step
- **Researcher → Halt**: If credible_sources < 2
- **Researcher → Analyst**: If credible_sources >= 2
- **Analyst → Writer**: When analysis complete
- **Writer → Supervisor**: For final review
- **Supervisor → END**: When brief meets criteria

## 5. Key Implementation Details

### Shared Memory Management
- Use LangGraph's built-in state persistence
- Each agent updates specific state fields
- State carries forward through all transitions

### Credibility Scoring
- Domain authority checks
- Publication type validation
- Recency verification
- Custom scoring algorithm

### Quality Gates
- Supervisor validates each stage output
- Retry logic for insufficient results
- Clear success/failure criteria

## 6. Sample Code Structure

```python
from langgraph.graph import StateGraph, END

# Define the workflow
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("supervisor", supervisor_agent)
workflow.add_node("researcher", researcher_agent)
workflow.add_node("analyst", analyst_agent)  
workflow.add_node("writer", writer_agent)

# Add edges with conditions
workflow.add_edge("supervisor", "researcher")
workflow.add_conditional_edges(
    "researcher",
    route_after_research,  # Function to check credibility
    {
        "continue": "analyst",
        "halt": END
    }
)
workflow.add_edge("analyst", "writer")
workflow.add_edge("writer", END)

# Set entry point
workflow.set_entry_point("supervisor")
```

## 7. Testing Strategy

Start with a simple test case:
1. Mock the web search results
2. Verify state transitions
3. Test halt conditions
4. Validate final output format

## 8. Implementation Checklist

- [ ] Set up LangGraph environment
- [ ] Define AgentState schema
- [ ] Implement Supervisor agent
- [ ] Implement Researcher agent with web search
- [ ] Add credibility scoring system
- [ ] Implement Analyst agent
- [ ] Implement Writer agent
- [ ] Set up conditional routing logic
- [ ] Add halt mechanism for weak sources
- [ ] Test with sample newsletter tool comparison
- [ ] Validate output format (table + citations)

## 9. Expected Output Format

The final brief should include:
- Executive summary
- Comparison table with pricing/features
- Pros and cons for each tool
- Recommendation with reasoning
- Source citations

## Next Steps

1. Choose specific LLM models for each agent
2. Set up web search integration (Tavily, SerpAPI, etc.)
3. Define credibility scoring criteria
4. Create output templates for consistent formatting