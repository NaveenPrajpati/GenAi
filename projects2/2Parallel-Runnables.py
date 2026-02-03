"""
Parallel Runnables - Run Multiple LLM Calls Concurrently
=========================================================

This module demonstrates running multiple LLM calls in parallel using
RunnableParallel (also known as RunnableMap).

Requirements Met:
-----------------
✅ Uses RunnableMap/RunnableParallel
✅ Single LLM instance (shared across all branches)
✅ No agent
✅ Generates summary, example, and interview question in parallel

Architecture:
-------------
    Input: {"topic": "Machine Learning"}
                    │
                    ▼
            ┌───────────────────┐
            │  RunnableParallel │
            │    (RunnableMap)  │
            └───────────────────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
        ▼           ▼           ▼
    ┌───────┐   ┌───────┐   ┌───────┐
    │Summary│   │Example│   │ Q&A   │
    │ Chain │   │ Chain │   │ Chain │
    └───────┘   └───────┘   └───────┘
        │           │           │
        └───────────┼───────────┘
                    │
                    ▼
            ┌───────────────────┐
            │   Combined Output │
            │  (All 3 results)  │
            └───────────────────┘

Note: RunnableParallel is the same as RunnableMap
      Both run branches concurrently for better performance

Expected Output:
----------------
{
  "summary": "...",
  "example": "...",
  "interview_question": "..."
}

Updated for LangChain 1.0+ (2025-2026)
"""

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import time

load_dotenv()


# =============================================================================
# Single LLM Instance (Shared Across All Parallel Branches)
# =============================================================================
# Using one model instance is more efficient - it shares connection pools

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()


# =============================================================================
# Define Individual Prompts for Each Parallel Branch
# =============================================================================

summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical educator. Be concise and clear."),
    ("human", "Provide a 2-3 sentence summary of '{topic}'.")
])

example_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical educator with industry experience."),
    ("human", "Give one real-world example of '{topic}' being used in practice. Be specific and practical.")
])

interview_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical interviewer. Create challenging but fair questions."),
    ("human", "Create one interview question about '{topic}' that tests deep understanding. Include what a good answer should cover.")
])


# =============================================================================
# Method 1: RunnableParallel with Dict Syntax
# =============================================================================
# This is the most common and readable approach

parallel_pipeline_dict = RunnableParallel({
    "summary": summary_prompt | model | parser,
    "example": example_prompt | model | parser,
    "interview_question": interview_prompt | model | parser,
})


# =============================================================================
# Method 2: RunnableParallel with Keyword Arguments
# =============================================================================
# Alternative syntax - same functionality

parallel_pipeline_kwargs = RunnableParallel(
    summary=summary_prompt | model | parser,
    example=example_prompt | model | parser,
    interview_question=interview_prompt | model | parser,
)


# =============================================================================
# Method 3: With RunnablePassthrough to Preserve Input
# =============================================================================
# Useful when you need to keep original input alongside generated content

parallel_with_input = RunnableParallel({
    "original_topic": RunnablePassthrough(),
    "summary": summary_prompt | model | parser,
    "example": example_prompt | model | parser,
    "interview_question": interview_prompt | model | parser,
})


# =============================================================================
# Comparison: Sequential vs Parallel Execution
# =============================================================================

def run_sequential(topic: str) -> dict:
    """Run chains one after another (slower)"""
    inputs = {"topic": topic}
    return {
        "summary": (summary_prompt | model | parser).invoke(inputs),
        "example": (example_prompt | model | parser).invoke(inputs),
        "interview_question": (interview_prompt | model | parser).invoke(inputs),
    }


def run_parallel(topic: str) -> dict:
    """Run all chains concurrently (faster)"""
    return parallel_pipeline_dict.invoke({"topic": topic})


def compare_performance(topic: str):
    """Demonstrate the performance difference"""
    print("\n" + "=" * 60)
    print("Performance Comparison: Sequential vs Parallel")
    print("=" * 60)

    # Sequential timing
    start = time.time()
    seq_result = run_sequential(topic)
    seq_time = time.time() - start
    print(f"\n⏱️  Sequential execution: {seq_time:.2f} seconds")

    # Parallel timing
    start = time.time()
    par_result = run_parallel(topic)
    par_time = time.time() - start
    print(f"⚡ Parallel execution: {par_time:.2f} seconds")

    # Speedup
    speedup = seq_time / par_time if par_time > 0 else 0
    print(f"🚀 Speedup: {speedup:.1f}x faster")

    return par_result


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("PARALLEL RUNNABLES - Concurrent LLM Calls Demo")
    print("=" * 60)

    topic = "Machine Learning"

    # ---------------------------------------------------------------------
    # Example 1: Basic Parallel Execution
    # ---------------------------------------------------------------------
    print(f"\n📌 Generating content for topic: '{topic}'")
    print("-" * 40)

    result = parallel_pipeline_dict.invoke({"topic": topic})

    print("\n✅ Expected Output Format:")
    print(f"""{{
  "summary": "...",
  "example": "...",
  "interview_question": "..."
}}""")

    print("\n📖 Summary:")
    print(f"   {result['summary']}")

    print("\n🎯 Real-World Example:")
    print(f"   {result['example']}")

    print("\n❓ Interview Question:")
    print(f"   {result['interview_question']}")

    # ---------------------------------------------------------------------
    # Example 2: With Original Input Preserved
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📌 With RunnablePassthrough (preserving input)")
    print("-" * 40)

    result_with_input = parallel_with_input.invoke({"topic": "Docker Containers"})
    print(f"Original Topic: {result_with_input['original_topic']}")
    print(f"Summary: {result_with_input['summary'][:100]}...")

    # ---------------------------------------------------------------------
    # Example 3: Batch Processing Multiple Topics
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📌 Batch Processing Multiple Topics")
    print("-" * 40)

    topics = ["REST APIs", "Microservices", "Kubernetes"]
    batch_inputs = [{"topic": t} for t in topics]

    batch_results = parallel_pipeline_dict.batch(batch_inputs)

    for i, res in enumerate(batch_results):
        print(f"\n--- {topics[i]} ---")
        print(f"Summary: {res['summary'][:80]}...")

    # ---------------------------------------------------------------------
    # Example 4: Performance Comparison
    # ---------------------------------------------------------------------
    compare_result = compare_performance("Neural Networks")


# =============================================================================
# Summary: RunnableParallel (RunnableMap) Key Points
# =============================================================================
"""
RunnableParallel / RunnableMap:
-------------------------------
Both names refer to the same class - runs multiple runnables concurrently.

Syntax Options:
---------------
1. Dict syntax:
   RunnableParallel({"key1": chain1, "key2": chain2})

2. Kwargs syntax:
   RunnableParallel(key1=chain1, key2=chain2)

Key Benefits:
-------------
- ⚡ Concurrent execution (typically 2-3x faster than sequential)
- 🔄 Shared model instance (efficient resource usage)
- 📦 Clean output structure (dict with named keys)
- 🔗 Composable (can be piped with other runnables)

When to Use:
------------
- Multiple independent LLM calls for same input
- Generating different perspectives on same topic
- Multi-output chains (summary + analysis + recommendations)
- Any scenario where operations don't depend on each other

Pro Tips:
---------
- Always use single model instance (avoids connection overhead)
- Add RunnablePassthrough() to preserve original input
- Use .batch() for processing multiple topics
- Combine with .stream() for real-time output (returns as each completes)
"""
