"""
Parallel Chain - Concurrent Execution
=====================================

LEARNING OBJECTIVES:
- Run multiple chains simultaneously for better performance
- Merge results from parallel chains
- Use different models in parallel

CONCEPT:
RunnableParallel executes multiple chains concurrently, then combines
their outputs into a single dictionary. This is useful when:
- You need multiple independent pieces of information
- You want to use different models for different tasks
- You need to speed up processing

VISUAL REPRESENTATION:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Parallel Chain Flow                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │                    {"text": "react.js"}                      │
    │                           │                                  │
    │              ┌────────────┴────────────┐                    │
    │              ▼                         ▼                     │
    │        ┌──────────┐              ┌──────────┐               │
    │        │ Prompt1  │              │ Prompt2  │               │
    │        │ (notes)  │              │  (quiz)  │               │
    │        └────┬─────┘              └────┬─────┘               │
    │             ▼                         ▼                      │
    │        ┌──────────┐              ┌──────────┐               │
    │        │ Gemini   │              │  GPT-4   │               │
    │        └────┬─────┘              └────┬─────┘               │
    │             │                         │                      │
    │             └────────────┬────────────┘                     │
    │                          ▼                                   │
    │                  {"notes": ..., "quiz": ...}                │
    │                          │                                   │
    │                          ▼                                   │
    │                    ┌──────────┐                             │
    │                    │ Prompt3  │                             │
    │                    │ (merge)  │                             │
    │                    └────┬─────┘                             │
    │                         ▼                                    │
    │                    Final Output                              │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

KEY INSIGHT:
RunnableParallel returns a dictionary with keys you define.
These keys can then be used in subsequent prompts:
    {"notes": ..., "quiz": ...} → Prompt3 uses {notes} and {quiz}

USE CASES:
1. Generate multiple types of content from same input
2. Use specialized models for different tasks
3. Reduce latency by parallelizing API calls
4. A/B test different models

PREREQUISITES:
- Completed: chains/simpleChain.py, chains/sequentialChain.py
- OpenAI and Google API keys in .env

NEXT STEPS:
- chains/conditionalChain.py - Branch based on conditions
- runnables/runnableParallel.py - More parallel patterns
"""

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# STEP 1: Define Parallel Prompts
# =============================================================================
# These prompts will run simultaneously

prompt_notes = PromptTemplate(
    template="Generate short and simple study notes on the topic: {text}\n\n"
             "Include key concepts, definitions, and important points.",
    input_variables=["text"]
)

prompt_quiz = PromptTemplate(
    template="Generate 5 short question-and-answer pairs from the following topic:\n\n"
             "Topic: {text}\n\n"
             "Format each as:\nQ: [question]\nA: [answer]",
    input_variables=["text"]
)

# Merge prompt uses outputs from both parallel chains
prompt_merge = PromptTemplate(
    template="Merge the following notes and quiz into a comprehensive study document:\n\n"
             "NOTES:\n{notes}\n\n"
             "QUIZ:\n{quiz}\n\n"
             "Create a well-formatted document with sections for Notes and Practice Questions.",
    input_variables=["notes", "quiz"]
)

# =============================================================================
# STEP 2: Initialize Multiple Models
# =============================================================================
# Using different models for different tasks demonstrates flexibility

# Google's Gemini for note generation (fast and capable)
model_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7
)

# OpenAI's GPT for quiz generation (great at structured output)
model_gpt = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.7
)

# Parser for text extraction
parser = StrOutputParser()

# =============================================================================
# STEP 3: Create Parallel Chain
# =============================================================================
# RunnableParallel takes a dictionary where:
# - Keys become the output dictionary keys
# - Values are the chains to run in parallel

parallel_chain = RunnableParallel({
    "notes": prompt_notes | model_gemini | parser,
    "quiz": prompt_quiz | model_gpt | parser
})

# Alternative syntax using keyword arguments:
# parallel_chain = RunnableParallel(
#     notes=prompt_notes | model_gemini | parser,
#     quiz=prompt_quiz | model_gpt | parser
# )

# =============================================================================
# STEP 4: Create Merge Chain
# =============================================================================
# This chain takes the parallel outputs and combines them
merge_chain = prompt_merge | model_gemini | parser

# =============================================================================
# STEP 5: Combine into Full Pipeline
# =============================================================================
# Parallel chain runs first, then merge chain combines results
full_chain = parallel_chain | merge_chain

# =============================================================================
# STEP 6: Execute and Display
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("PARALLEL CHAIN DEMO")
    print("=" * 70)
    print("\nTopic: React.js")
    print("-" * 70)

    # Run the full chain
    result = full_chain.invoke({"text": "react.js"})

    print("\nFINAL MERGED OUTPUT:")
    print("-" * 70)
    print(result)
    print("=" * 70)

    # Visualize chain structure
    print("\nCHAIN STRUCTURE:")
    full_chain.get_graph().print_ascii()

    # ==========================================================================
    # DEBUGGING: Inspect parallel outputs separately
    # ==========================================================================
    print("\n" + "=" * 70)
    print("DEBUGGING: Inspect Parallel Outputs")
    print("=" * 70)

    # Run just the parallel part to see individual outputs
    parallel_output = parallel_chain.invoke({"text": "Python programming"})

    print("\nNotes Output (from Gemini):")
    print("-" * 70)
    print(parallel_output["notes"][:500] + "..." if len(parallel_output["notes"]) > 500 else parallel_output["notes"])

    print("\nQuiz Output (from GPT-4o-mini):")
    print("-" * 70)
    print(parallel_output["quiz"])

    # ==========================================================================
    # ADVANCED: Parallel with passthrough
    # ==========================================================================
    from langchain_core.runnables import RunnablePassthrough

    print("\n" + "=" * 70)
    print("ADVANCED: Parallel with Passthrough")
    print("=" * 70)

    # Include original input alongside parallel outputs
    chain_with_passthrough = RunnableParallel({
        "original_topic": RunnablePassthrough(),  # Pass input through unchanged
        "notes": prompt_notes | model_gemini | parser,
        "quiz": prompt_quiz | model_gpt | parser
    })

    result_with_original = chain_with_passthrough.invoke({"text": "JavaScript"})
    print(f"\nOriginal input preserved: {result_with_original['original_topic']}")
    print(f"Notes generated: {len(result_with_original['notes'])} characters")
    print(f"Quiz generated: {len(result_with_original['quiz'])} characters")
