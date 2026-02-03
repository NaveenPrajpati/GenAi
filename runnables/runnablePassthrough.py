"""
RunnablePassthrough - Identity & Data Forwarding in LCEL
=========================================================

RunnablePassthrough is a utility component that passes input through unchanged.
It's essential for preserving data while other branches transform it.

Key Concepts:
-------------
1. Identity Function: Returns exactly what it receives
2. Data Preservation: Keep original input alongside transformed outputs
3. RunnablePassthrough.assign(): Add new keys while keeping existing ones

Common Use Cases:
-----------------
- RAG: Pass question through while retrieving context
- Parallel execution: Keep original input in one branch
- Data enrichment: Add computed fields to existing data

LCEL Pattern:
-------------
    RunnablePassthrough()           -> passes input unchanged
    RunnablePassthrough.assign(     -> adds new keys to input dict
        new_key=some_runnable
    )

Architecture:
-------------
    Input: {"topic": "AI"}
           │
           ├──► RunnablePassthrough() ──► {"topic": "AI"} (unchanged)
           │
           └──► chain ──► "transformed output"

Updated for LangChain 1.0+ (2025-2026)
"""

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from dotenv import load_dotenv

load_dotenv()


# =============================================================================
# Example 1: Basic RunnablePassthrough
# =============================================================================
# RunnablePassthrough passes input through unchanged
# Useful when you want to keep original data alongside transformed data

prompt = PromptTemplate.from_template("Generate a LinkedIn post about {topic}")

model = ChatOpenAI(model="gpt-4o-mini")
parser = StrOutputParser()

# Parallel chain: one branch passes through, other transforms
parallel_chain = RunnableParallel({
    "original_input": RunnablePassthrough(),  # Passes {"topic": "..."} unchanged
    "linkedin_post": prompt | model | parser   # Transforms to LinkedIn post
})

print("=" * 60)
print("Example 1: Basic RunnablePassthrough")
print("=" * 60)
result = parallel_chain.invoke({"topic": "brain"})
print(f"Original Input: {result['original_input']}")
print(f"LinkedIn Post: {result['linkedin_post'][:200]}...")


# =============================================================================
# Example 2: RunnablePassthrough.assign() - Add Keys to Dict
# =============================================================================
# assign() keeps all existing keys and adds new computed ones
# This is extremely useful for building up context incrementally

def get_word_count(input_dict: dict) -> int:
    """Count words in topic"""
    return len(input_dict.get("topic", "").split())

# Chain that adds computed fields while preserving original
enrichment_chain = RunnablePassthrough.assign(
    word_count=lambda x: get_word_count(x),
    uppercase_topic=lambda x: x["topic"].upper()
)

print("\n" + "=" * 60)
print("Example 2: RunnablePassthrough.assign()")
print("=" * 60)
enriched = enrichment_chain.invoke({"topic": "artificial intelligence"})
print(f"Enriched data: {enriched}")
# Output: {"topic": "artificial intelligence", "word_count": 2, "uppercase_topic": "ARTIFICIAL INTELLIGENCE"}


# =============================================================================
# Example 3: RAG Pattern with RunnablePassthrough
# =============================================================================
# The most common use case: pass question through while retrieving context

def mock_retriever(input_dict: dict) -> str:
    """Simulates document retrieval - in real use, this would be a vector store"""
    topic = input_dict.get("question", "")
    return f"Retrieved context about: {topic}. Key facts: This is sample context."

rag_prompt = PromptTemplate.from_template(
    """Answer the question based on the context.

Context: {context}

Question: {question}

Answer:"""
)

# RAG chain using assign to add context while keeping question
rag_chain = (
    RunnablePassthrough.assign(
        context=lambda x: mock_retriever(x)  # Add context, keep question
    )
    | rag_prompt
    | model
    | parser
)

print("\n" + "=" * 60)
print("Example 3: RAG Pattern with RunnablePassthrough")
print("=" * 60)
rag_result = rag_chain.invoke({"question": "What is machine learning?"})
print(f"RAG Response: {rag_result[:300]}...")


# =============================================================================
# Example 4: Combining with RunnableParallel
# =============================================================================
# Use passthrough in parallel branches for complex data flows

summary_prompt = PromptTemplate.from_template("Summarize this in one sentence: {topic}")
expand_prompt = PromptTemplate.from_template("Explain {topic} in detail for beginners")

multi_output_chain = RunnableParallel({
    "original": RunnablePassthrough(),           # Keep original input
    "summary": summary_prompt | model | parser,   # Generate summary
    "detailed": expand_prompt | model | parser    # Generate detailed explanation
})

print("\n" + "=" * 60)
print("Example 4: Multi-Output with Passthrough")
print("=" * 60)
multi_result = multi_output_chain.invoke({"topic": "neural networks"})
print(f"Original: {multi_result['original']}")
print(f"Summary: {multi_result['summary'][:150]}...")
print(f"Detailed: {multi_result['detailed'][:150]}...")


# =============================================================================
# Example 5: Chaining Multiple assign() Calls
# =============================================================================
# Build up context step by step

def analyze_sentiment(text: str) -> str:
    """Mock sentiment analysis"""
    return "positive" if any(word in text.lower() for word in ["good", "great", "happy"]) else "neutral"

multi_step_chain = (
    RunnablePassthrough.assign(
        word_count=lambda x: len(x["text"].split())
    )
    .assign(
        sentiment=lambda x: analyze_sentiment(x["text"])
    )
    .assign(
        summary=lambda x: f"Text with {x['word_count']} words, sentiment: {x['sentiment']}"
    )
)

print("\n" + "=" * 60)
print("Example 5: Chained assign() Calls")
print("=" * 60)
chained_result = multi_step_chain.invoke({"text": "This is a great day for learning!"})
print(f"Step-by-step enrichment: {chained_result}")


# =============================================================================
# Summary: When to Use RunnablePassthrough
# =============================================================================
"""
Use RunnablePassthrough when you need to:

1. PRESERVE INPUT: Keep original data in parallel with transformations
   RunnableParallel({
       "original": RunnablePassthrough(),
       "transformed": some_chain
   })

2. BUILD RAG CHAINS: Pass question while adding retrieved context
   RunnablePassthrough.assign(context=retriever) | prompt | model

3. ENRICH DATA: Add computed fields without losing existing ones
   RunnablePassthrough.assign(
       new_field=lambda x: compute(x)
   )

4. CHAIN ENRICHMENTS: Build up data incrementally
   RunnablePassthrough.assign(a=...).assign(b=...).assign(c=...)

Key Differences:
- RunnablePassthrough() → Returns input unchanged
- RunnablePassthrough.assign() → Returns input + new keys
- RunnableLambda() → Transforms input arbitrarily

Pro Tips:
- Use assign() instead of manual dict merging
- Combine with RunnableParallel for complex flows
- Perfect for RAG where you need both question and context
"""