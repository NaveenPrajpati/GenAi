"""
Simple Chain - The Foundation of LangChain
==========================================

LEARNING OBJECTIVES:
- Understand the basic Prompt → Model → Parser pipeline
- Learn LCEL (LangChain Expression Language) pipe operator
- Create your first working chain

CONCEPT:
A "chain" in LangChain is a sequence of components that process data.
The simplest chain has three parts:

    PromptTemplate → ChatModel → OutputParser
         |              |            |
    Formats input   Generates    Extracts
    into prompt     response     text output

The pipe operator (|) connects these components using LCEL.

LCEL (LangChain Expression Language):
- Declarative way to compose components
- Supports streaming, async, batching out of the box
- Components connected with | operator

EXAMPLE FLOW:
    Input: {"topic": "plants"}
           ↓
    PromptTemplate: "Generate 5 interesting facts about plants"
           ↓
    ChatModel: Sends to LLM API
           ↓
    StrOutputParser: Extracts string content
           ↓
    Output: "1. Plants produce oxygen... 2. ..."

PREREQUISITES:
- OpenAI API key in .env file
- pip install langchain-openai langchain-core python-dotenv

NEXT STEPS:
- chains/sequentialChain.py - Chain multiple operations
- chains/parallelChain.py - Run chains in parallel
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

# Load environment variables (OPENAI_API_KEY)
load_dotenv()

# =============================================================================
# STEP 1: Create a Prompt Template
# =============================================================================
# PromptTemplate formats your input into a proper prompt string.
# Variables in curly braces {} are placeholders for user input.

prompt = PromptTemplate(
    template="Generate 5 interesting facts about {topic}",
    input_variables=["topic"]  # List of expected input keys
)

# You can also use the shorthand:
# prompt = PromptTemplate.from_template("Generate 5 interesting facts about {topic}")

# =============================================================================
# STEP 2: Initialize the Language Model
# =============================================================================
# ChatOpenAI is a wrapper around OpenAI's chat models.
# Common parameters:
#   - model: Model name (gpt-4o-mini, gpt-4o, gpt-4-turbo)
#   - temperature: Creativity (0=deterministic, 1=creative)
#   - max_tokens: Maximum response length

model = ChatOpenAI(
    model="gpt-4o-mini",  # Cost-effective and fast model
    temperature=0.7       # Balanced creativity
)

# =============================================================================
# STEP 3: Create an Output Parser
# =============================================================================
# StrOutputParser extracts the string content from the model's response.
# The model returns a complex Message object; the parser extracts just the text.

parser = StrOutputParser()

# =============================================================================
# STEP 4: Compose the Chain with LCEL
# =============================================================================
# The pipe operator (|) connects components left-to-right.
# Data flows: prompt → model → parser

chain = prompt | model | parser

# =============================================================================
# STEP 5: Invoke the Chain
# =============================================================================
# invoke() runs the chain with the given input dictionary.
# The keys must match the input_variables in your prompt.

if __name__ == "__main__":
    # Run the chain
    result = chain.invoke({"topic": "plants"})
    print("=" * 60)
    print("SIMPLE CHAIN OUTPUT")
    print("=" * 60)
    print(result)
    print("=" * 60)

    # Visualize the chain structure
    print("\nCHAIN STRUCTURE:")
    chain.get_graph().print_ascii()

    # ==========================================================================
    # BONUS: Other ways to run chains
    # ==========================================================================

    # Batch processing - run multiple inputs at once
    print("\n" + "=" * 60)
    print("BATCH PROCESSING")
    print("=" * 60)
    batch_results = chain.batch([
        {"topic": "oceans"},
        {"topic": "mountains"}
    ])
    for i, res in enumerate(batch_results):
        print(f"\nResult {i+1}:")
        print(res[:200] + "..." if len(res) > 200 else res)

    # Streaming - get output token by token
    print("\n" + "=" * 60)
    print("STREAMING OUTPUT")
    print("=" * 60)
    print("Streaming facts about space:")
    for chunk in chain.stream({"topic": "space"}):
        print(chunk, end="", flush=True)
    print()
