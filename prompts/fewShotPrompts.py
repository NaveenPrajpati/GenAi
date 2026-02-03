"""
Few-Shot Prompting - Learning from Examples
=============================================

LEARNING OBJECTIVES:
- Use examples to guide LLM behavior
- Create FewShotPromptTemplate for consistent formatting
- Implement dynamic example selection
- Build effective example sets

CONCEPT:
Few-shot prompting provides the LLM with examples of the desired
input-output behavior. The LLM learns the pattern and applies it
to new inputs. This is one of the most effective prompting techniques.

FEW-SHOT LEARNING:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Few-Shot Learning                         │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  Zero-shot: No examples                                     │
    │  "Translate 'hello' to French"                              │
    │                                                              │
    │  One-shot: One example                                      │
    │  "dog -> chien                                              │
    │   cat -> ?"                                                 │
    │                                                              │
    │  Few-shot: Multiple examples (2-5 typically)                │
    │  "dog -> chien                                              │
    │   cat -> chat                                               │
    │   bird -> oiseau                                            │
    │   house -> ?"                                               │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

WHEN TO USE FEW-SHOT:
    ┌──────────────────────┬─────────────────────────────────────┐
    │ Use Case             │ Why Few-Shot Helps                  │
    ├──────────────────────┼─────────────────────────────────────┤
    │ Format specification │ Shows exact output structure        │
    │ Classification       │ Demonstrates categories             │
    │ Style transfer       │ Shows desired tone/style            │
    │ Data extraction      │ Shows what to extract               │
    │ Translation          │ Shows domain-specific terms         │
    │ Code generation      │ Shows coding style/patterns         │
    └──────────────────────┴─────────────────────────────────────┘

PREREQUISITES:
- Completed: prompts/promptTemplates.py
- Completed: prompts/chatPrompts.py

NEXT STEPS:
- chains/conditionalChain.py - Use with classification
- structure-output/withStructureOutput.py - Structured outputs
"""

from langchain_core.prompts import (
    FewShotPromptTemplate,
    FewShotChatMessagePromptTemplate,
    PromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.prompts.example_selector import (
    SemanticSimilarityExampleSelector,
    LengthBasedExampleSelector,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# 1. BASIC FEW-SHOT TEMPLATE
# =============================================================================
print("=" * 70)
print("1. BASIC FEW-SHOT TEMPLATE")
print("=" * 70)
print("""
USE CASE: Simple pattern learning with consistent format
""")

# Define examples
examples = [
    {"word": "happy", "antonym": "sad"},
    {"word": "tall", "antonym": "short"},
    {"word": "fast", "antonym": "slow"},
]

# Template for each example
example_template = PromptTemplate(
    input_variables=["word", "antonym"],
    template="Word: {word}\nAntonym: {antonym}"
)

# Create few-shot prompt
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="Give the antonym of every word.\n",
    suffix="\nWord: {input}\nAntonym:",
    input_variables=["input"]
)

# Format the prompt
formatted = few_shot_prompt.format(input="hot")
print(f"\nFew-shot prompt:\n{formatted}")

# =============================================================================
# 2. FEW-SHOT WITH CHAT MODELS
# =============================================================================
print("\n" + "=" * 70)
print("2. FEW-SHOT WITH CHAT MODELS")
print("=" * 70)
print("""
USE CASE: Examples in chat format for GPT-4, Claude, etc.
""")

# Examples as conversations
chat_examples = [
    {"input": "What's 2+2?", "output": "2+2 equals 4."},
    {"input": "What's the capital of France?", "output": "The capital of France is Paris."},
    {"input": "Who wrote Romeo and Juliet?", "output": "William Shakespeare wrote Romeo and Juliet."},
]

# Template for each example (as a mini-conversation)
example_prompt = ChatPromptTemplate.from_messages([
    ("human", "{input}"),
    ("ai", "{output}")
])

# Create few-shot chat prompt
few_shot_chat = FewShotChatMessagePromptTemplate(
    examples=chat_examples,
    example_prompt=example_prompt,
)

# Full prompt with system message
full_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer questions clearly and concisely."),
    few_shot_chat,
    ("human", "{question}")
])

messages = full_chat_prompt.format_messages(question="What's the tallest mountain?")
print(f"\nTotal messages: {len(messages)}")
for m in messages:
    preview = m.content[:40] + "..." if len(m.content) > 40 else m.content
    print(f"  [{m.type}]: {preview}")

# =============================================================================
# 3. CLASSIFICATION WITH FEW-SHOT
# =============================================================================
print("\n" + "=" * 70)
print("3. CLASSIFICATION WITH FEW-SHOT")
print("=" * 70)
print("""
USE CASE: Sentiment analysis, intent detection, categorization
""")

classification_examples = [
    {
        "text": "I absolutely love this product! Best purchase ever!",
        "sentiment": "POSITIVE"
    },
    {
        "text": "This is terrible. Complete waste of money.",
        "sentiment": "NEGATIVE"
    },
    {
        "text": "It's okay, nothing special but gets the job done.",
        "sentiment": "NEUTRAL"
    },
    {
        "text": "Amazing quality and fast shipping!",
        "sentiment": "POSITIVE"
    },
    {
        "text": "Worst customer service I've ever experienced.",
        "sentiment": "NEGATIVE"
    }
]

classification_template = PromptTemplate(
    input_variables=["text", "sentiment"],
    template="Text: {text}\nSentiment: {sentiment}"
)

sentiment_prompt = FewShotPromptTemplate(
    examples=classification_examples,
    example_prompt=classification_template,
    prefix="""Classify the sentiment of the text as POSITIVE, NEGATIVE, or NEUTRAL.

Examples:
""",
    suffix="\nText: {input}\nSentiment:",
    input_variables=["input"]
)

formatted = sentiment_prompt.format(
    input="The quality is decent but the price is too high."
)
print(f"\nClassification prompt:\n{formatted}")

# =============================================================================
# 4. DATA EXTRACTION WITH FEW-SHOT
# =============================================================================
print("\n" + "=" * 70)
print("4. DATA EXTRACTION WITH FEW-SHOT")
print("=" * 70)
print("""
USE CASE: Extracting structured data from unstructured text
""")

extraction_examples = [
    {
        "text": "John Smith, 35 years old, works as a software engineer at Google in San Francisco.",
        "extracted": "Name: John Smith\nAge: 35\nOccupation: Software Engineer\nCompany: Google\nLocation: San Francisco"
    },
    {
        "text": "Dr. Sarah Johnson is a 42-year-old cardiologist at Mayo Clinic, Rochester.",
        "extracted": "Name: Sarah Johnson\nAge: 42\nOccupation: Cardiologist\nCompany: Mayo Clinic\nLocation: Rochester"
    },
    {
        "text": "Mike Chen, age 28, is a data scientist working remotely for Amazon.",
        "extracted": "Name: Mike Chen\nAge: 28\nOccupation: Data Scientist\nCompany: Amazon\nLocation: Remote"
    }
]

extraction_template = PromptTemplate(
    input_variables=["text", "extracted"],
    template="Text: {text}\nExtracted Information:\n{extracted}"
)

extraction_prompt = FewShotPromptTemplate(
    examples=extraction_examples,
    example_prompt=extraction_template,
    prefix="Extract person information from the text in the following format:\n\n",
    suffix="\nText: {input}\nExtracted Information:",
    input_variables=["input"]
)

formatted = extraction_prompt.format(
    input="Lisa Wang, 31, leads the AI research team at Microsoft in Seattle."
)
print(f"\nExtraction prompt:\n{formatted}")

# =============================================================================
# 5. CODE GENERATION WITH FEW-SHOT
# =============================================================================
print("\n" + "=" * 70)
print("5. CODE GENERATION WITH FEW-SHOT")
print("=" * 70)
print("""
USE CASE: Teaching coding patterns, function generation
""")

code_examples = [
    {
        "description": "A function that adds two numbers",
        "code": '''def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b'''
    },
    {
        "description": "A function that checks if a number is even",
        "code": '''def is_even(n: int) -> bool:
    """Check if a number is even."""
    return n % 2 == 0'''
    },
    {
        "description": "A function that reverses a string",
        "code": '''def reverse_string(s: str) -> str:
    """Reverse a string and return it."""
    return s[::-1]'''
    }
]

code_template = PromptTemplate(
    input_variables=["description", "code"],
    template="Description: {description}\nCode:\n```python\n{code}\n```"
)

code_prompt = FewShotPromptTemplate(
    examples=code_examples,
    example_prompt=code_template,
    prefix="""Generate Python functions based on the description.
Follow these conventions:
- Include type hints
- Add docstrings
- Keep functions simple and focused

Examples:
""",
    suffix="\nDescription: {input}\nCode:",
    input_variables=["input"]
)

formatted = code_prompt.format(
    input="A function that calculates the factorial of a number"
)
print(f"\nCode generation prompt:\n{formatted}")

# =============================================================================
# 6. STYLE TRANSFER WITH FEW-SHOT
# =============================================================================
print("\n" + "=" * 70)
print("6. STYLE TRANSFER WITH FEW-SHOT")
print("=" * 70)
print("""
USE CASE: Converting text to a specific style or tone
""")

style_examples = [
    {
        "original": "The meeting is at 3pm.",
        "casual": "Hey! We're meeting up at 3 - see you there! 😊"
    },
    {
        "original": "Please complete the report.",
        "casual": "Could you wrap up that report when you get a chance? Thanks!"
    },
    {
        "original": "The project deadline has been extended.",
        "casual": "Good news - we've got more time on the project! 🎉"
    }
]

style_template = PromptTemplate(
    input_variables=["original", "casual"],
    template="Formal: {original}\nCasual: {casual}"
)

style_prompt = FewShotPromptTemplate(
    examples=style_examples,
    example_prompt=style_template,
    prefix="Convert formal text to casual, friendly messages:\n\n",
    suffix="\nFormal: {input}\nCasual:",
    input_variables=["input"]
)

formatted = style_prompt.format(
    input="Your request has been processed successfully."
)
print(f"\nStyle transfer prompt:\n{formatted}")

# =============================================================================
# 7. DYNAMIC EXAMPLE SELECTION
# =============================================================================
print("\n" + "=" * 70)
print("7. DYNAMIC EXAMPLE SELECTION")
print("=" * 70)
print("""
USE CASE: Choose relevant examples based on input
""")

# Large pool of examples
math_examples = [
    {"problem": "2 + 3", "solution": "5", "type": "addition"},
    {"problem": "10 - 4", "solution": "6", "type": "subtraction"},
    {"problem": "5 * 6", "solution": "30", "type": "multiplication"},
    {"problem": "20 / 4", "solution": "5", "type": "division"},
    {"problem": "15 + 27", "solution": "42", "type": "addition"},
    {"problem": "100 - 35", "solution": "65", "type": "subtraction"},
    {"problem": "8 * 9", "solution": "72", "type": "multiplication"},
    {"problem": "144 / 12", "solution": "12", "type": "division"},
]

# Length-based selection (fits within token limit)
length_selector = LengthBasedExampleSelector(
    examples=math_examples,
    example_prompt=PromptTemplate(
        input_variables=["problem", "solution"],
        template="Problem: {problem}\nSolution: {solution}"
    ),
    max_length=100  # Max characters
)

# Create prompt with dynamic selection
dynamic_prompt = FewShotPromptTemplate(
    example_selector=length_selector,
    example_prompt=PromptTemplate(
        input_variables=["problem", "solution"],
        template="Problem: {problem}\nSolution: {solution}"
    ),
    prefix="Solve the math problem:\n",
    suffix="\nProblem: {input}\nSolution:",
    input_variables=["input"]
)

formatted = dynamic_prompt.format(input="25 + 17")
print(f"\nDynamic selection prompt:\n{formatted}")

# =============================================================================
# 8. SEMANTIC SIMILARITY SELECTION
# =============================================================================
print("\n" + "=" * 70)
print("8. SEMANTIC SIMILARITY SELECTION")
print("=" * 70)
print("""
USE CASE: Select most relevant examples based on meaning
(Requires embeddings - shown as concept)
""")

# Note: This requires embeddings setup
print("""
# Example code for semantic selection:

from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

# Examples with different topics
topic_examples = [
    {"question": "How do I cook pasta?", "answer": "Boil water, add pasta..."},
    {"question": "How do I bake a cake?", "answer": "Mix flour, eggs..."},
    {"question": "How do I write a Python function?", "answer": "Use def keyword..."},
    {"question": "How do I create a class in Java?", "answer": "Use class keyword..."},
]

# Create semantic selector
embeddings = OpenAIEmbeddings()
semantic_selector = SemanticSimilarityExampleSelector.from_examples(
    topic_examples,
    embeddings,
    Chroma,
    k=2  # Select 2 most similar examples
)

# Query about cooking -> selects cooking examples
# Query about coding -> selects coding examples
""")

# =============================================================================
# 9. MULTI-STEP REASONING WITH FEW-SHOT
# =============================================================================
print("\n" + "=" * 70)
print("9. MULTI-STEP REASONING (Chain-of-Thought)")
print("=" * 70)
print("""
USE CASE: Complex problems requiring step-by-step reasoning
""")

cot_examples = [
    {
        "question": "If John has 5 apples and gives 2 to Mary, how many does he have left?",
        "reasoning": """Let's think step by step:
1. John starts with 5 apples
2. He gives away 2 apples
3. 5 - 2 = 3
Therefore, John has 3 apples left."""
    },
    {
        "question": "A train travels 60 miles in 2 hours. What is its speed?",
        "reasoning": """Let's think step by step:
1. Distance traveled = 60 miles
2. Time taken = 2 hours
3. Speed = Distance / Time
4. Speed = 60 / 2 = 30 miles per hour
Therefore, the train's speed is 30 mph."""
    }
]

cot_template = PromptTemplate(
    input_variables=["question", "reasoning"],
    template="Question: {question}\n{reasoning}"
)

cot_prompt = FewShotPromptTemplate(
    examples=cot_examples,
    example_prompt=cot_template,
    prefix="Solve problems using step-by-step reasoning:\n\n",
    suffix="\nQuestion: {input}\nLet's think step by step:",
    input_variables=["input"]
)

formatted = cot_prompt.format(
    input="If a book costs $15 and you have $50, how many books can you buy?"
)
print(f"\nChain-of-thought prompt:\n{formatted}")

# =============================================================================
# 10. USING FEW-SHOT WITH CHAINS
# =============================================================================
print("\n" + "=" * 70)
print("10. FEW-SHOT IN CHAINS")
print("=" * 70)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
parser = StrOutputParser()

# Create chain with few-shot
sentiment_chain = sentiment_prompt | model | parser

print("\nFew-shot chain created:")
print("  sentiment_prompt -> model -> parser")
print("  Ready for classification tasks!")

# =============================================================================
# BEST PRACTICES
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("FEW-SHOT PROMPTING BEST PRACTICES")
    print("=" * 70)
    print("""
    1. NUMBER OF EXAMPLES:
       - 2-5 examples is usually optimal
       - More examples = more tokens = higher cost
       - Diminishing returns after 5-6 examples

    2. EXAMPLE QUALITY:
       - Use diverse, representative examples
       - Cover edge cases if relevant
       - Ensure examples are correct!

    3. EXAMPLE ORDER:
       - Put most relevant examples last (recency bias)
       - Or use semantic selection for dynamic ordering

    4. CONSISTENCY:
       - Keep format identical across examples
       - Use the same structure for input and output

    5. CHAIN-OF-THOUGHT:
       - For complex tasks, show reasoning process
       - "Let's think step by step" triggers better reasoning

    6. EXAMPLE SELECTION:
       - Use SemanticSimilarityExampleSelector for diverse topics
       - Use LengthBasedExampleSelector for token limits

    7. TESTING:
       - Test with inputs similar to AND different from examples
       - Verify the model generalizes, not just memorizes

    8. DYNAMIC EXAMPLES:
       - Consider selecting examples based on input
       - Cache frequently used example combinations
    """)
