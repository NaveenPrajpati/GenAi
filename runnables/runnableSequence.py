"""
RunnableSequence - Sequential Composition
==========================================

LEARNING OBJECTIVES:
- Understand RunnableSequence as the building block of LCEL
- Compose components sequentially with the pipe operator
- Debug and inspect sequence execution
- Handle data flow between steps

CONCEPT:
RunnableSequence chains multiple runnables together, where the output
of one becomes the input of the next. It's the foundation of LCEL
(LangChain Expression Language).

WAYS TO CREATE SEQUENCES:
    ┌─────────────────────────────────────────────────────────────┐
    │              RunnableSequence Creation Methods               │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Method 1: Pipe Operator (|)                                │
    │  chain = prompt | model | parser                            │
    │                                                              │
    │  Method 2: Explicit Constructor                             │
    │  chain = RunnableSequence(prompt, model, parser)            │
    │                                                              │
    │  Method 3: Using .pipe() method                             │
    │  chain = prompt.pipe(model).pipe(parser)                    │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

DATA FLOW:
    Input Dict → [Prompt] → String → [Model] → AIMessage → [Parser] → Output
         │           │          │         │           │           │
         └───────────┴──────────┴─────────┴───────────┴───────────┘
                              Data flows left to right

RUNNABLE INTERFACE:
All runnables share these methods:
- invoke(): Single input, sync
- ainvoke(): Single input, async
- batch(): Multiple inputs, sync
- abatch(): Multiple inputs, async
- stream(): Stream output chunks
- astream(): Async stream

PREREQUISITES:
- Completed: chains/simpleChain.py
- Understanding of Python type hints

NEXT STEPS:
- runnables/runnableParallel.py - Parallel execution
- runnables/runnableLambda.py - Custom functions
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# STEP 1: Create Components
# =============================================================================
print("=" * 70)
print("RUNNABLE SEQUENCE DEMO")
print("=" * 70)

# Prompt template
prompt = PromptTemplate(
    template="Write a joke about {topic}",
    input_variables=["topic"]
)

# Model
model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Parser
parser = StrOutputParser()

# =============================================================================
# STEP 2: Create Sequence (Multiple Ways)
# =============================================================================
print("\n--- Method 1: Pipe Operator ---")
chain_pipe = prompt | model | parser
print(f"Type: {type(chain_pipe).__name__}")

print("\n--- Method 2: Explicit Constructor ---")
chain_explicit = RunnableSequence(prompt, model, parser)
print(f"Type: {type(chain_explicit).__name__}")

# =============================================================================
# STEP 3: Invoke the Chain
# =============================================================================
print("\n" + "=" * 70)
print("INVOKING THE CHAIN")
print("=" * 70)

result = chain_pipe.invoke({"topic": "computer"})
print(f"\nResult: {result}")

# =============================================================================
# STEP 4: Batch Processing
# =============================================================================
print("\n" + "=" * 70)
print("BATCH PROCESSING")
print("=" * 70)

batch_results = chain_pipe.batch([
    {"topic": "cats"},
    {"topic": "dogs"},
    {"topic": "robots"}
])

for i, res in enumerate(batch_results):
    print(f"\n{i+1}. {res[:100]}...")

# =============================================================================
# STEP 5: Streaming
# =============================================================================
print("\n" + "=" * 70)
print("STREAMING OUTPUT")
print("=" * 70)

print("\nStreaming: ", end="")
for chunk in chain_pipe.stream({"topic": "coffee"}):
    print(chunk, end="", flush=True)
print()

# =============================================================================
# STEP 6: Visualize Chain
# =============================================================================
print("\n" + "=" * 70)
print("CHAIN VISUALIZATION")
print("=" * 70)

chain_pipe.get_graph().print_ascii()

# =============================================================================
# BEST PRACTICES
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("RUNNABLE SEQUENCE BEST PRACTICES")
    print("=" * 70)
    print("""
    1. Use pipe operator (|) for readability
    2. Each component must be a Runnable
    3. Output of step N must match input of step N+1
    4. Use .get_graph().print_ascii() to visualize
    5. Use batch() over loops for multiple inputs
    6. Use stream() for long-running operations
    """)
