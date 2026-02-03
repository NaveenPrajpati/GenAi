# LangChain & LangGraph Complete Tutorial

A comprehensive, hands-on tutorial covering LangChain and LangGraph from fundamentals to production-ready patterns.

## Prerequisites

- Python 3.10+
- Basic understanding of LLMs and prompts
- OpenAI API key (or other provider keys)

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install langchain langchain-core langchain-openai langgraph langchain-community
pip install chromadb faiss-cpu python-dotenv streamlit

# Set up environment variables
cp .env.example .env
# Add your API keys to .env
```

## Learning Path

### Level 1: Foundations (Start Here)

Understanding the basic building blocks of LangChain.

| Order | File | Concept | Description |
|-------|------|---------|-------------|
| 1.1 | `chains/simpleChain.py` | Basic Chain | Prompt → Model → Parser pipeline |
| 1.2 | `chains/sequentialChain.py` | Sequential Chains | Chaining multiple operations |
| 1.3 | `chains/parallelChain.py` | Parallel Chains | Running chains concurrently |
| 1.4 | `chains/conditionalChain.py` | Conditional Chains | Branching based on conditions |

### Level 2: Prompts & Templates

Mastering prompt engineering patterns for effective LLM interaction.

| Order | File | Concept | Description |
|-------|------|---------|-------------|
| 2.1 | `prompts/promptTemplates.py` | Basic Templates | PromptTemplate, ChatPromptTemplate, composition |
| 2.2 | `prompts/chatPrompts.py` | Chat Prompts | Message roles, history, personas, RAG context |
| 2.3 | `prompts/fewShotPrompts.py` | Few-Shot Learning | Examples, dynamic selection, chain-of-thought |

### Level 3: Runnables (Composition Patterns)

Deep dive into LangChain Expression Language (LCEL) and Runnable composition.

| Order | File | Concept | Description |
|-------|------|---------|-------------|
| 3.1 | `runnables/runnableSequence.py` | RunnableSequence | Sequential composition with pipe |
| 3.2 | `runnables/runnableParallel.py` | RunnableParallel | Parallel execution |
| 3.3 | `runnables/runnableLambda.py` | RunnableLambda | Custom Python functions |
| 3.4 | `runnables/runnablePassthrough.py` | RunnablePassthrough | Identity and forwarding |
| 3.5 | `runnables/runnableBranch.py` | RunnableBranch | Conditional routing |

### Level 4: Tools & Agents

Creating and using tools with LLMs.

| Order | File | Concept | Description |
|-------|------|---------|-------------|
| 4.1 | `tools/customTools.py` | Tool Creation | Three ways to create tools |
| 4.2 | `tools/toolbinding.py` | Tool Binding | Connecting tools to LLMs |
| 4.3 | `tools/toolbinding2.py` | Advanced Binding | Tool invocation patterns |
| 4.4 | `tools/toolkit.py` | Toolkits | Managing tool collections |
| 4.5 | `tools/buitinTools.py` | Built-in Tools | Using pre-made tools |

### Level 5: Document Loading & Text Processing

Ingesting and processing various document types.

| Order | File | Concept | Description |
|-------|------|---------|-------------|
| 5.1 | `documentLoader/textLoader.py` | Text Loading | Plain text files |
| 5.2 | `documentLoader/pdfLoader.py` | PDF Loading | PDF documents |
| 5.3 | `documentLoader/csvLoader.py` | CSV Loading | Tabular data |
| 5.4 | `documentLoader/webBaseLoader.py` | Web Loading | Web pages |
| 5.5 | `documentLoader/directoryLoader.py` | Directory Loading | Batch loading |
| 5.6 | `textSpliter/lengthBased.py` | Length Splitting | Character-based chunks |
| 5.7 | `textSpliter/recursiveSpliter.py` | Recursive Splitting | Hierarchical chunks |

### Level 6: Retrieval & RAG

Building retrieval-augmented generation systems.

| Order | File | Concept | Description |
|-------|------|---------|-------------|
| 6.1 | `retriver/retriver.py` | Basic Retrieval | Wikipedia retrieval |
| 6.2 | `retriver/vectorstoreRetriever.py` | Vector Store | Similarity search |
| 6.3 | `retriver/contextualCompressionRetriever.py` | Compression | Context optimization |
| 6.4 | `RAG/semanticSearch.py` | Semantic Search | Vector embeddings |
| 6.5 | `RAG/youtubeChatbot.py` | RAG Chatbot | Complete RAG system |

### Level 7: Structured Output

Getting structured data from LLMs.

| Order | File | Concept | Description |
|-------|------|---------|-------------|
| 7.1 | `structure-output/typedict.py` | TypedDict | Type-hinted outputs |
| 7.2 | `structure-output/withStructureOutput.py` | Structured Output | Pydantic validation |

### Level 8: LangGraph Fundamentals

Introduction to stateful graph workflows.

| Order | File | Concept | Description |
|-------|------|---------|-------------|
| 8.1 | `langgraph/first.ipynb` | Graph Basics | StateGraph introduction |
| 8.2 | `langgraph/bmiWorkflow.ipynb` | Simple Workflow | BMI calculator graph |
| 8.3 | `langgraph/conditionalWorkflow.py` | Conditional Edges | Branching in graphs |
| 8.4 | `langgraph/parallelWorkflow.ipynb` | Parallel Nodes | Concurrent execution |
| 8.5 | `langgraph/llmWorkflow.ipynb` | LLM Nodes | ChatModel integration |

### Level 9: LangGraph Advanced

Production-ready patterns and features.

| Order | File | Concept | Description |
|-------|------|---------|-------------|
| 9.1 | `langgraph/persistance.py` | Checkpointing | State persistence |
| 9.2 | `langgraph/persistance2.py` | Time Travel | State history & resume |
| 9.3 | `langgraph/faultTolerance.py` | Error Handling | Retry policies |
| 9.4 | `langgraph/basicChatbot.py` | Chatbot | Multi-thread chat |
| 9.5 | `langgraph/fullFeatureChatbot.py` | Production Chat | Full-featured UI |
| 9.6 | `langgraph/streaming.py` | Streaming | Real-time output |
| 9.7 | `langgraph/humanInTheLoop.py` | Human-in-Loop | interrupt() pattern |
| 9.8 | `langgraph/commandPattern.py` | Dynamic Routing | Command-based flow |

### Level 10: Practice Projects

Complete working applications.

| Order | File | Concept | Description |
|-------|------|---------|-------------|
| 10.1 | `practicsProjects/basicAgentComponents.py` | Components | 11 essential parts |
| 10.2 | `practicsProjects/1tool-use.py` | Tool Agent | Guardrails & safety |
| 10.3 | `practicsProjects/2ragPdf.py` | RAG Pipeline | PDF Q&A system |
| 10.4 | `practicsProjects/3webSearch.py` | Web Research | Search workflow |
| 10.5 | `practicsProjects/4dataAnalystAgent.py` | Data Analysis | Analytics agent |
| 10.6 | `practicsProjects/5triageAgent.py` | Triage | Request routing |
| 10.7 | `practicsProjects/6multiAgent.py` | Multi-Agent | Team coordination |

### Level 11: Production & Observability

Monitoring, debugging, and deployment.

| Order | File | Concept | Description |
|-------|------|---------|-------------|
| 11.1 | `observability/langsmith_tracing.py` | LangSmith | Tracing & debugging |
| 11.2 | `observability/evaluation.py` | Evaluation | Testing agents |

---

## Core Concepts Reference

### LangChain Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                     LangChain Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │  Prompt  │ -> │  Model   │ -> │  Parser  │ -> │  Output  │  │
│  │ Template │    │  (LLM)   │    │          │    │          │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                                  │
│  Composition: prompt | model | parser  (LCEL pipe operator)     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### LangGraph State Machine

```
┌─────────────────────────────────────────────────────────────────┐
│                     LangGraph Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│     ┌───────┐                                                   │
│     │ START │                                                   │
│     └───┬───┘                                                   │
│         │                                                       │
│         ▼                                                       │
│     ┌───────┐      ┌───────┐      ┌───────┐                    │
│     │ Node1 │ ---> │ Node2 │ ---> │ Node3 │                    │
│     └───────┘      └───┬───┘      └───────┘                    │
│                        │                                        │
│                        │ (conditional edge)                     │
│                        ▼                                        │
│                    ┌───────┐                                    │
│                    │ Tools │ ◄─── (loop back)                   │
│                    └───────┘                                    │
│                        │                                        │
│                        ▼                                        │
│                    ┌───────┐                                    │
│                    │  END  │                                    │
│                    └───────┘                                    │
│                                                                  │
│  Key: State flows through nodes, edges control routing          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### The 11 Essential LangGraph Components

1. **State** - Shared data structure (TypedDict with reducers)
2. **Nodes** - Processing units (Python functions)
3. **Edges** - Flow control (sequential or conditional)
4. **Compile** - Convert builder to executable graph
5. **Tools & ToolNode** - External actions/APIs
6. **LLMs & Prompts** - AI model integration
7. **Memory & Persistence** - Checkpointers for state
8. **Human-in-the-Loop** - interrupt() for approvals
9. **Retries & Caching** - Resilience patterns
10. **Streaming** - Real-time updates
11. **Observability** - LangSmith integration

---

## Quick Reference

### Creating a Simple Chain
```python
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = PromptTemplate.from_template("Tell me about {topic}")
model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

chain = prompt | model | parser
result = chain.invoke({"topic": "AI"})
```

### Creating a LangGraph Agent
```python
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import InMemorySaver

graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_edge(START, "agent")
graph.add_edge("agent", END)

app = graph.compile(checkpointer=InMemorySaver())
result = app.invoke({"messages": [("user", "Hello")]})
```

### Creating a Tool
```python
from langchain_core.tools import tool

@tool
def search(query: str) -> str:
    """Search for information about a topic."""
    return f"Results for: {query}"
```

### Prompt Templates Quick Reference
```python
# Basic PromptTemplate
from langchain_core.prompts import PromptTemplate
prompt = PromptTemplate.from_template("Tell me about {topic}")

# ChatPromptTemplate (for chat models)
from langchain_core.prompts import ChatPromptTemplate
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    ("human", "{question}")
])

# Few-Shot Template
from langchain_core.prompts import FewShotPromptTemplate
examples = [{"input": "2+2", "output": "4"}]
few_shot = FewShotPromptTemplate(
    examples=examples,
    example_prompt=PromptTemplate.from_template("Input: {input}\nOutput: {output}"),
    prefix="Solve math problems:",
    suffix="Input: {input}\nOutput:",
    input_variables=["input"]
)

# With MessagesPlaceholder (conversation history)
from langchain_core.prompts import MessagesPlaceholder
history_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])
```

---

## Environment Variables

Create a `.env` file with:

```env
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
ANTHROPIC_API_KEY=...
TAVILY_API_KEY=...
LANGCHAIN_API_KEY=...
LANGCHAIN_TRACING_V2=true
```

---

## Resources

- [LangChain Documentation](https://python.langchain.com/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith](https://smith.langchain.com/)
- [LangChain GitHub](https://github.com/langchain-ai/langchain)
- [LangGraph GitHub](https://github.com/langchain-ai/langgraph)

---

## Version Compatibility

This tutorial is compatible with:
- LangChain 1.0+
- LangGraph 1.0+
- Python 3.10+

Last updated: February 2026
