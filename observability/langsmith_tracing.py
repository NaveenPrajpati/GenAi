"""
LangSmith Tracing & Observability
==================================

LEARNING OBJECTIVES:
- Set up LangSmith for tracing LangChain applications
- Understand traces, spans, and runs
- Debug chains and agents with trace data
- Evaluate LLM outputs with custom metrics

CONCEPT:
LangSmith is the observability platform for LangChain applications.
It allows you to:
1. Trace every step of your chains and agents
2. Debug issues by inspecting inputs/outputs at each step
3. Evaluate LLM performance with custom metrics
4. Monitor production applications

TRACING ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │                   LangSmith Tracing                          │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │   Your Application                    LangSmith Cloud       │
    │   ┌─────────────┐                    ┌─────────────────┐    │
    │   │   Chain     │ ─── traces ───────►│   Dashboard     │    │
    │   │   Agent     │                    │   - Runs        │    │
    │   │   Tools     │                    │   - Latency     │    │
    │   └─────────────┘                    │   - Errors      │    │
    │                                      │   - Costs       │    │
    │   Each step creates a "run":         └─────────────────┘    │
    │   ┌─────────────────────────────────────────────────┐       │
    │   │ Run (parent)                                    │       │
    │   │  ├─ Run (LLM call)                             │       │
    │   │  │   ├─ input: "Hello"                         │       │
    │   │  │   ├─ output: "Hi there!"                    │       │
    │   │  │   └─ latency: 0.5s                          │       │
    │   │  ├─ Run (Tool call)                            │       │
    │   │  │   └─ ...                                    │       │
    │   │  └─ Run (Final output)                         │       │
    │   └─────────────────────────────────────────────────┘       │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

SETUP REQUIREMENTS:
1. Create account at https://smith.langchain.com
2. Get API key from settings
3. Set environment variables (see below)

PREREQUISITES:
- LangSmith account
- LANGCHAIN_API_KEY in .env
- Understanding of chains and agents

NEXT STEPS:
- observability/evaluation.py - LLM evaluation
- Production monitoring patterns
"""

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.tracers import LangChainTracer
from langsmith import Client

load_dotenv()

# =============================================================================
# STEP 1: Configure LangSmith
# =============================================================================
# These environment variables enable automatic tracing

# Option 1: Set via environment (recommended for production)
# Add to your .env file:
# LANGCHAIN_TRACING_V2=true
# LANGCHAIN_API_KEY=your-api-key
# LANGCHAIN_PROJECT=my-project-name

# Option 2: Set programmatically
os.environ["LANGCHAIN_TRACING_V2"] = "true"
# os.environ["LANGCHAIN_API_KEY"] = "your-key"  # Better to use .env
os.environ.setdefault("LANGCHAIN_PROJECT", "langchain-tutorial")

print("=" * 70)
print("LANGSMITH CONFIGURATION")
print("=" * 70)
print(f"Tracing enabled: {os.environ.get('LANGCHAIN_TRACING_V2', 'false')}")
print(f"Project: {os.environ.get('LANGCHAIN_PROJECT', 'default')}")
print(f"API Key configured: {'Yes' if os.environ.get('LANGCHAIN_API_KEY') else 'No'}")


# =============================================================================
# STEP 2: Basic Tracing (Automatic)
# =============================================================================
# When LANGCHAIN_TRACING_V2=true, all LangChain operations are traced automatically

print("\n" + "=" * 70)
print("STEP 2: Basic Automatic Tracing")
print("=" * 70)

# Create a simple chain
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
prompt = PromptTemplate.from_template("Tell me a {adjective} joke about {topic}")
chain = prompt | model | StrOutputParser()

# This call is automatically traced!
print("\nRunning chain (automatically traced)...")
result = chain.invoke({"adjective": "funny", "topic": "programming"})
print(f"Result: {result[:100]}...")
print("\n→ Check LangSmith dashboard to see the trace!")


# =============================================================================
# STEP 3: Custom Run Names and Tags
# =============================================================================
# Add metadata to make traces more searchable

print("\n" + "=" * 70)
print("STEP 3: Custom Run Names and Tags")
print("=" * 70)

# Add custom metadata to the chain invocation
result_with_tags = chain.invoke(
    {"adjective": "clever", "topic": "AI"},
    config={
        "run_name": "joke-generator",  # Custom name for this run
        "tags": ["demo", "joke", "v1"],  # Tags for filtering
        "metadata": {  # Custom metadata
            "user_id": "demo-user",
            "version": "1.0",
            "experiment": "tutorial"
        }
    }
)
print(f"Result: {result_with_tags[:100]}...")
print("\n→ Find this run by tag 'demo' in LangSmith!")


# =============================================================================
# STEP 4: Explicit Tracer Configuration
# =============================================================================
# Use LangChainTracer for more control

print("\n" + "=" * 70)
print("STEP 4: Explicit Tracer")
print("=" * 70)

# Create tracer with specific project
tracer = LangChainTracer(project_name="tutorial-experiments")

# Use tracer in chain call
result_explicit = chain.invoke(
    {"adjective": "witty", "topic": "coffee"},
    config={"callbacks": [tracer]}
)
print(f"Result: {result_explicit[:100]}...")
print("\n→ Check 'tutorial-experiments' project in LangSmith!")


# =============================================================================
# STEP 5: Tracing Custom Functions
# =============================================================================
# Trace your own Python functions

print("\n" + "=" * 70)
print("STEP 5: Tracing Custom Functions")
print("=" * 70)

from langsmith import traceable


@traceable(name="process_text")
def process_text(text: str) -> str:
    """Custom function that will be traced."""
    # Your processing logic here
    words = text.split()
    processed = " ".join(w.upper() for w in words)
    return processed


@traceable(name="analyze_sentiment")
def analyze_sentiment(text: str) -> dict:
    """Another traced function."""
    # Simplified sentiment analysis
    positive_words = {"good", "great", "excellent", "happy", "love"}
    negative_words = {"bad", "terrible", "sad", "hate", "awful"}

    words = set(text.lower().split())
    pos_count = len(words & positive_words)
    neg_count = len(words & negative_words)

    return {
        "text": text,
        "positive_score": pos_count,
        "negative_score": neg_count,
        "sentiment": "positive" if pos_count > neg_count else "negative" if neg_count > pos_count else "neutral"
    }


# These function calls are now traced
result1 = process_text("hello world from langsmith")
result2 = analyze_sentiment("This is a great and excellent product!")
print(f"Processed: {result1}")
print(f"Sentiment: {result2}")


# =============================================================================
# STEP 6: Nested Tracing
# =============================================================================
# Create hierarchical traces with parent-child relationships

print("\n" + "=" * 70)
print("STEP 6: Nested Tracing")
print("=" * 70)


@traceable(name="full_analysis_pipeline")
def full_analysis_pipeline(text: str) -> dict:
    """
    A traced pipeline with nested function calls.
    Each function creates a child span in the trace.
    """
    # Step 1: Process text (creates child span)
    processed = process_text(text)

    # Step 2: Generate description using LLM (creates child span)
    description_chain = (
        PromptTemplate.from_template("Describe this in one sentence: {text}")
        | model
        | StrOutputParser()
    )
    description = description_chain.invoke({"text": processed})

    # Step 3: Analyze sentiment (creates child span)
    sentiment = analyze_sentiment(description)

    return {
        "original": text,
        "processed": processed,
        "description": description,
        "sentiment": sentiment
    }


# Run the pipeline - creates a nested trace
pipeline_result = full_analysis_pipeline("LangSmith makes debugging easy")
print(f"Pipeline result: {pipeline_result['sentiment']['sentiment']}")
print("\n→ See the nested spans in LangSmith!")


# =============================================================================
# STEP 7: LangSmith Client for Programmatic Access
# =============================================================================
# Use the client to access runs programmatically

print("\n" + "=" * 70)
print("STEP 7: LangSmith Client")
print("=" * 70)

# Check if API key is configured
if os.environ.get("LANGCHAIN_API_KEY"):
    try:
        client = Client()

        # List recent runs
        print("\nRecent runs in project:")
        project_name = os.environ.get("LANGCHAIN_PROJECT", "default")
        runs = list(client.list_runs(project_name=project_name, limit=5))

        for run in runs:
            print(f"  - {run.name}: {run.status} ({run.run_type})")

    except Exception as e:
        print(f"Could not connect to LangSmith: {e}")
        print("Make sure LANGCHAIN_API_KEY is set correctly.")
else:
    print("LANGCHAIN_API_KEY not set - skipping client demo")
    print("Set the key to access run data programmatically")


# =============================================================================
# STEP 8: Error Tracing
# =============================================================================
# Errors are automatically captured in traces

print("\n" + "=" * 70)
print("STEP 8: Error Tracing")
print("=" * 70)


@traceable(name="risky_operation")
def risky_operation(value: int) -> int:
    """Operation that might fail."""
    if value < 0:
        raise ValueError(f"Value must be positive, got {value}")
    return value * 2


# This will be traced with the error
try:
    risky_operation(-5)
except ValueError as e:
    print(f"Caught error (traced): {e}")
    print("\n→ Error details visible in LangSmith!")


# =============================================================================
# BEST PRACTICES
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("LANGSMITH BEST PRACTICES")
    print("=" * 70)
    print("""
    1. ENVIRONMENT SETUP:
       - Set LANGCHAIN_TRACING_V2=true in production
       - Use separate projects for dev/staging/prod
       - Never commit API keys to git

    2. MEANINGFUL NAMES:
       - Use run_name for custom operation names
       - Add tags for filtering (e.g., ["prod", "user-facing"])
       - Include metadata (user_id, version, etc.)

    3. COST TRACKING:
       - Token usage is automatically tracked
       - Set up cost alerts in LangSmith
       - Use metadata to group costs by feature

    4. DEBUGGING:
       - Check latency per step
       - Compare inputs/outputs at each stage
       - Use tags to find problematic runs

    5. EVALUATION:
       - Create datasets from production runs
       - Run automated evaluations
       - Compare model versions

    6. PRODUCTION:
       - Use async tracing for minimal overhead
       - Set up alerts for errors and latency
       - Monitor token usage trends

    VIEW YOUR TRACES:
    → https://smith.langchain.com
    """)
