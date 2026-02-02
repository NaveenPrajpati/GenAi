"""
RunnableParallel - Concurrent Execution
========================================

LEARNING OBJECTIVES:
- Execute multiple chains concurrently
- Combine results from parallel operations
- Use passthrough for preserving input
- Optimize latency with parallel API calls

CONCEPT:
RunnableParallel runs multiple runnables at the same time, then
combines their outputs into a dictionary. This is useful when:
- You need to make multiple independent LLM calls
- You want to reduce total latency
- Different processing paths converge into one

PARALLEL EXECUTION:
    ┌─────────────────────────────────────────────────────────────┐
    │                   RunnableParallel Flow                      │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │   Input: {"topic": "AI"}                                    │
    │              │                                               │
    │    ┌─────────┴─────────┐                                    │
    │    │                   │                                     │
    │    ▼                   ▼                                     │
    │ ┌──────┐           ┌──────┐                                 │
    │ │ joke │           │ post │   (run in parallel)             │
    │ │chain │           │chain │                                 │
    │ └──┬───┘           └──┬───┘                                 │
    │    │                   │                                     │
    │    └─────────┬─────────┘                                    │
    │              ▼                                               │
    │   Output: {"joke": "...", "linkedin": "..."}                │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

PREREQUISITES:
- Completed: runnables/runnableSequence.py

NEXT STEPS:
- runnables/runnableLambda.py - Custom functions
- runnables/runnableBranch.py - Conditional routing
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableSequence, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# STEP 1: Create Prompts
# =============================================================================
print("=" * 70)
print("RUNNABLE PARALLEL DEMO")
print("=" * 70)

prompt_joke = PromptTemplate(
    template="Write a short, funny joke about {topic}",
    input_variables=["topic"]
)

prompt_linkedin = PromptTemplate(
    template="Generate a professional LinkedIn post about {topic}",
    input_variables=["topic"]
)

# =============================================================================
# STEP 2: Create Model and Parser
# =============================================================================
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

# =============================================================================
# STEP 3: Create Parallel Chain
# =============================================================================
chain = RunnableParallel({
    "joke": RunnableSequence(prompt_joke, model, parser),
    "linkedin": RunnableSequence(prompt_linkedin, model, parser)
})

# =============================================================================
# STEP 4: Invoke and Display
# =============================================================================
if __name__ == "__main__":
    result = chain.invoke({"topic": "brain"})
    
    print("\n--- Joke ---")
    print(result["joke"])
    
    print("\n--- LinkedIn Post ---")
    print(result["linkedin"])
    
    print("\n--- Chain Structure ---")
    chain.get_graph().print_ascii()
