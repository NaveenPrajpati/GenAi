"""
Runnable Foundations - Pure LCEL Pipeline (No LLMChain)
========================================================

This module demonstrates building a pure Runnable pipeline using modern
LangChain Expression Language (LCEL) patterns.

Requirements Met:
-----------------
✅ ChatPromptTemplate
✅ ChatOpenAI
✅ Runnable composition (|)
✅ No deprecated APIs (using langchain_core)
✅ Supports .invoke(), .batch(), .stream()
✅ RunnablePassthrough to preserve input (Bonus)

Architecture:
-------------
    Input: {"topic": "AI"}
           │
           ├──► RunnablePassthrough ──► preserves original input
           │
           └──► RunnableParallel
                    ├──► definition_chain ──► "definition"
                    ├──► use_cases_chain  ──► "use_cases"
                    └──► pitfalls_chain   ──► "pitfalls"
                                │
                                ▼
                    Structured JSON Output

Expected Output:
----------------
{
  "original_input": {"topic": "..."},
  "definition": "...",
  "use_cases": ["...", "..."],
  "pitfalls": ["..."]
}

Updated for LangChain 1.0+ (2025-2026)
"""

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import List
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Pydantic Models for Structured Output
# =============================================================================

class TopicExplanation(BaseModel):
    """Structured output schema for topic explanation"""
    definition: str = Field(description="Clear definition of the topic")
    use_cases: List[str] = Field(description="List of real-world use cases")
    pitfalls: List[str] = Field(description="Common pitfalls or limitations")


# =============================================================================
# Initialize Chat Model
# =============================================================================

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()


# =============================================================================
# Approach 1: Parallel Chains with String Outputs
# =============================================================================
# Runs all three prompts in parallel for efficiency

definition_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical educator. Provide clear, concise explanations."),
    ("human", "Provide a clear, one-paragraph definition of '{topic}'.")
])

use_cases_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical educator. List practical applications."),
    ("human", "List exactly 3 real-world use cases of '{topic}'. Format as a numbered list.")
])

pitfalls_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a technical educator. Identify common mistakes and limitations."),
    ("human", "List exactly 3 common pitfalls or limitations of '{topic}'. Format as a numbered list.")
])

# Individual chains using pipe operator (|)
definition_chain = definition_prompt | model | parser
use_cases_chain = use_cases_prompt | model | parser
pitfalls_chain = pitfalls_prompt | model | parser

# Parallel pipeline with RunnablePassthrough to preserve original input
parallel_pipeline = RunnableParallel({
    "original_input": RunnablePassthrough(),  # Bonus: preserves input
    "definition": definition_chain,
    "use_cases": use_cases_chain,
    "pitfalls": pitfalls_chain,
})


# =============================================================================
# Approach 2: Single Prompt with Structured JSON Output
# =============================================================================
# More efficient: single LLM call returning structured data

structured_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a technical educator. Always respond with valid JSON.
Your response must follow this exact structure:
{{
  "definition": "A clear one-paragraph definition",
  "use_cases": ["use case 1", "use case 2", "use case 3"],
  "pitfalls": ["pitfall 1", "pitfall 2", "pitfall 3"]
}}"""),
    ("human", "Explain '{topic}' with its definition, 3 use cases, and 3 pitfalls.")
])

# Using JsonOutputParser for automatic JSON parsing
json_parser = JsonOutputParser(pydantic_object=TopicExplanation)

structured_pipeline = (
    RunnablePassthrough.assign(
        format_instructions=lambda _: json_parser.get_format_instructions()
    )
    | structured_prompt
    | model
    | json_parser
)

# Alternative: Using with_structured_output (most elegant approach)
structured_model = model.with_structured_output(TopicExplanation)

elegant_pipeline = (
    ChatPromptTemplate.from_messages([
        ("system", "You are a technical educator providing structured explanations."),
        ("human", "Explain '{topic}': provide a definition, 3 use cases, and 3 pitfalls.")
    ])
    | structured_model
)


# =============================================================================
# Pipeline Functions Supporting .invoke(), .batch(), .stream()
# =============================================================================

def invoke_example(topic: str) -> dict:
    """
    Demonstrates .invoke() - Single synchronous call
    """
    return parallel_pipeline.invoke({"topic": topic})


def batch_example(topics: List[str]) -> List[dict]:
    """
    Demonstrates .batch() - Process multiple inputs efficiently
    """
    inputs = [{"topic": t} for t in topics]
    return parallel_pipeline.batch(inputs)


def stream_example(topic: str):
    """
    Demonstrates .stream() - Real-time streaming output
    Note: Streaming works best with single-output chains
    """
    # For streaming, use a single chain (parallel doesn't stream well)
    single_chain = definition_prompt | model | parser

    print(f"\nStreaming definition for '{topic}':")
    print("-" * 40)
    for chunk in single_chain.stream({"topic": topic}):
        print(chunk, end="", flush=True)
    print("\n")


async def ainvoke_example(topic: str) -> dict:
    """
    Demonstrates .ainvoke() - Async invocation
    """
    return await parallel_pipeline.ainvoke({"topic": topic})


# =============================================================================
# Main Execution
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNABLE FOUNDATIONS - Pure LCEL Pipeline Demo")
    print("=" * 60)

    # ---------------------------------------------------------------------
    # Example 1: .invoke() - Single topic
    # ---------------------------------------------------------------------
    print("\n📌 Example 1: .invoke() - Single Topic")
    print("-" * 40)

    result = invoke_example("LangChain Runnables")

    print(f"\nOriginal Input: {result['original_input']}")
    print(f"\n📖 Definition:\n{result['definition']}")
    print(f"\n🎯 Use Cases:\n{result['use_cases']}")
    print(f"\n⚠️  Pitfalls:\n{result['pitfalls']}")

    # ---------------------------------------------------------------------
    # Example 2: .batch() - Multiple topics
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📌 Example 2: .batch() - Multiple Topics")
    print("-" * 40)

    topics = ["Python Decorators", "REST APIs"]
    batch_results = batch_example(topics)

    for i, res in enumerate(batch_results):
        print(f"\n--- Topic {i+1}: {topics[i]} ---")
        print(f"Definition: {res['definition'][:100]}...")

    # ---------------------------------------------------------------------
    # Example 3: .stream() - Real-time output
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📌 Example 3: .stream() - Real-time Streaming")

    stream_example("Machine Learning")

    # ---------------------------------------------------------------------
    # Example 4: Structured JSON Output
    # ---------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📌 Example 4: Structured JSON Output")
    print("-" * 40)

    structured_result = elegant_pipeline.invoke({"topic": "Docker Containers"})

    print(f"\nStructured Output (Pydantic Model):")
    print(f"  Definition: {structured_result.definition[:100]}...")
    print(f"  Use Cases: {structured_result.use_cases}")
    print(f"  Pitfalls: {structured_result.pitfalls}")

    # Convert to dict for JSON serialization
    print(f"\nAs JSON: {structured_result.model_dump()}")


# =============================================================================
# Summary: Key LCEL Patterns Used
# =============================================================================
"""
LCEL Composition Patterns:
--------------------------

1. PIPE OPERATOR (|) - Sequential composition
   chain = prompt | model | parser

2. RunnableParallel - Concurrent execution
   parallel = RunnableParallel({
       "key1": chain1,
       "key2": chain2
   })

3. RunnablePassthrough - Preserve input
   RunnablePassthrough()           # Returns input unchanged
   RunnablePassthrough.assign()    # Adds keys while keeping input

4. with_structured_output() - Type-safe JSON output
   model.with_structured_output(PydanticModel)

Supported Methods (all Runnables):
----------------------------------
- .invoke(input)        → Single sync call
- .ainvoke(input)       → Single async call
- .batch([inputs])      → Multiple sync calls
- .abatch([inputs])     → Multiple async calls
- .stream(input)        → Streaming output
- .astream(input)       → Async streaming

Pro Tips:
---------
- Use RunnableParallel for independent operations (3x faster than sequential)
- Use with_structured_output() for type-safe JSON responses
- Prefer .batch() over loops for multiple inputs (optimized batching)
- Stream single chains; parallel chains return all-at-once
"""
