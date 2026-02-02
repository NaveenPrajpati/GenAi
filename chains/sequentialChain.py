"""
Sequential Chain - Chaining Multiple Operations
================================================

LEARNING OBJECTIVES:
- Chain multiple LLM calls sequentially
- Pass output from one chain as input to the next
- Understand data flow between chain steps

CONCEPT:
A sequential chain runs multiple operations one after another,
where the output of one step becomes the input for the next.

EXAMPLE FLOW:
    Input: {"topic": "gen ai"}
           ↓
    Chain 1: Generate detailed report
           ↓
    Output: "Generative AI is a type of..."
           ↓
    Chain 2: Summarize into 5 points
           ↓
    Final Output: "1. GenAI creates content... 2. ..."

VISUAL REPRESENTATION:
    ┌─────────────────────────────────────────────────────────────┐
    │                   Sequential Chain Flow                      │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │   {"topic": "AI"}                                           │
    │         │                                                   │
    │         ▼                                                   │
    │   ┌──────────┐   ┌──────────┐   ┌──────────┐               │
    │   │ Prompt1  │ → │  Model   │ → │  Parser  │ ──┐           │
    │   │(report)  │   │          │   │          │   │           │
    │   └──────────┘   └──────────┘   └──────────┘   │           │
    │                                                 │ (text)    │
    │                                                 ▼           │
    │   ┌──────────┐   ┌──────────┐   ┌──────────┐               │
    │   │ Prompt2  │ → │  Model   │ → │  Parser  │ → Output      │
    │   │(summary) │   │          │   │          │               │
    │   └──────────┘   └──────────┘   └──────────┘               │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

KEY INSIGHT:
When chaining prompts, the output variable from the first chain
must match the input variable expected by the second prompt.

    Prompt1 output: string → becomes → Prompt2's {text} variable

PREREQUISITES:
- Completed: chains/simpleChain.py
- OpenAI API key in .env

NEXT STEPS:
- chains/parallelChain.py - Run chains in parallel
- chains/conditionalChain.py - Branch based on conditions
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# =============================================================================
# STEP 1: Define the Prompts
# =============================================================================
# First prompt: Generate detailed content
prompt1 = PromptTemplate(
    template="Generate a detailed report on {topic}. Include key concepts, "
             "history, current applications, and future prospects.",
    input_variables=["topic"]
)

# Second prompt: Summarize the content
# NOTE: The input variable {text} will receive the output from prompt1
prompt2 = PromptTemplate(
    template="Generate a 5-point summary from the following text:\n\n{text}",
    input_variables=["text"]
)

# =============================================================================
# STEP 2: Initialize Model and Parser
# =============================================================================
# Using gpt-4o-mini for cost-effective processing
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# StrOutputParser extracts text from model response
parser = StrOutputParser()

# =============================================================================
# STEP 3: Create the Sequential Chain
# =============================================================================
# Chain components together with the pipe operator
# The output of one step automatically becomes input for the next

chain = prompt1 | model | parser | prompt2 | model | parser
#       └──────── Step 1 ───────┘   └──────── Step 2 ───────┘

# How it works:
# 1. prompt1 receives {"topic": "gen ai"}
# 2. model generates detailed report
# 3. parser extracts text string
# 4. prompt2 receives this text as {text}
# 5. model generates summary
# 6. parser extracts final output

# =============================================================================
# STEP 4: Execute and Display
# =============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("SEQUENTIAL CHAIN DEMO")
    print("=" * 70)
    print("\nTopic: Generative AI")
    print("-" * 70)

    # Run the chain
    result = chain.invoke({"topic": "gen ai"})

    print("\nFINAL OUTPUT (5-Point Summary):")
    print("-" * 70)
    print(result)
    print("=" * 70)

    # Visualize chain structure
    print("\nCHAIN STRUCTURE:")
    chain.get_graph().print_ascii()

    # ==========================================================================
    # ALTERNATIVE: Using RunnableSequence explicitly
    # ==========================================================================
    from langchain_core.runnables import RunnableSequence

    # Same chain, but using explicit RunnableSequence
    step1 = prompt1 | model | parser
    step2 = prompt2 | model | parser
    explicit_chain = RunnableSequence(step1, step2)

    print("\n" + "=" * 70)
    print("ALTERNATIVE: Explicit RunnableSequence")
    print("=" * 70)
    result2 = explicit_chain.invoke({"topic": "machine learning"})
    print(result2)

    # ==========================================================================
    # DEBUGGING: Inspect intermediate steps
    # ==========================================================================
    print("\n" + "=" * 70)
    print("DEBUGGING: Inspect Intermediate Steps")
    print("=" * 70)

    # Run step 1 alone to see intermediate output
    step1_output = step1.invoke({"topic": "blockchain"})
    print("\nStep 1 Output (Detailed Report - first 500 chars):")
    print("-" * 70)
    print(step1_output[:500] + "...")

    # Run step 2 with step 1's output
    step2_output = step2.invoke({"text": step1_output})
    print("\nStep 2 Output (Summary):")
    print("-" * 70)
    print(step2_output)
