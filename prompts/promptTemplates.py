"""
Prompt Templates - The Foundation of LLM Communication
=======================================================

LEARNING OBJECTIVES:
- Understand different prompt template types
- Use PromptTemplate for simple string formatting
- Use ChatPromptTemplate for chat-based models
- Use FewShotPromptTemplate for examples-based prompting
- Apply partial variables and template composition

CONCEPT:
Prompt templates are reusable blueprints for creating prompts.
They separate the prompt structure from the dynamic content,
making your code more maintainable and flexible.

PROMPT TEMPLATE TYPES:
    ┌─────────────────────────────────────────────────────────────┐
    │                    Prompt Template Types                     │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  1. PromptTemplate                                          │
    │     - Simple string templates with {variables}              │
    │     - Best for: completion models, simple prompts           │
    │                                                              │
    │  2. ChatPromptTemplate                                      │
    │     - Templates with roles (system, human, ai)              │
    │     - Best for: chat models (GPT-4, Claude)                 │
    │                                                              │
    │  3. FewShotPromptTemplate                                   │
    │     - Include examples in the prompt                        │
    │     - Best for: teaching patterns, formatting               │
    │                                                              │
    │  4. MessagesPlaceholder                                     │
    │     - Insert dynamic message history                        │
    │     - Best for: chat with memory                            │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

USE CASE MAPPING:
    ┌──────────────────────┬─────────────────────────────────────┐
    │ Use Case             │ Best Template Type                  │
    ├──────────────────────┼─────────────────────────────────────┤
    │ Simple generation    │ PromptTemplate                      │
    │ Chat conversation    │ ChatPromptTemplate                  │
    │ Pattern learning     │ FewShotPromptTemplate               │
    │ Multi-turn chat      │ ChatPromptTemplate + Placeholder    │
    │ Structured output    │ ChatPromptTemplate + format instr   │
    │ Agent instructions   │ ChatPromptTemplate (system role)    │
    └──────────────────────┴─────────────────────────────────────┘

PREREQUISITES:
- Basic Python string formatting
- Understanding of chat model roles

NEXT STEPS:
- prompts/chatPrompts.py - Advanced chat templates
- prompts/fewShotPrompts.py - Few-shot learning
- chains/simpleChain.py - Use prompts in chains
"""

from langchain_core.prompts import (
    PromptTemplate,
    ChatPromptTemplate,
    FewShotPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# 1. BASIC PROMPT TEMPLATE
# =============================================================================
print("=" * 70)
print("1. BASIC PROMPT TEMPLATE")
print("=" * 70)
print("""
USE CASE: Simple text generation, completion tasks
- Single string with placeholders
- Good for straightforward prompts
""")

# Method 1: Using constructor
prompt1 = PromptTemplate(
    template="Write a {length} story about a {animal} who learns to {skill}.",
    input_variables=["length", "animal", "skill"]
)

# Method 2: Using from_template (auto-detects variables)
prompt2 = PromptTemplate.from_template(
    "Translate the following {source_lang} text to {target_lang}: {text}"
)

# Format the prompt
formatted = prompt1.format(length="short", animal="cat", skill="code")
print(f"\nFormatted prompt:\n{formatted}")

# Check input variables
print(f"\nInput variables: {prompt2.input_variables}")

# =============================================================================
# 2. CHAT PROMPT TEMPLATE
# =============================================================================
print("\n" + "=" * 70)
print("2. CHAT PROMPT TEMPLATE")
print("=" * 70)
print("""
USE CASE: Chat models (GPT-4, Claude), role-based conversations
- Defines system, human, and AI messages
- Maintains conversation structure
""")

# Method 1: Using from_messages (most common)
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant specialized in {domain}."),
    ("human", "{user_input}")
])

# Method 2: Using message template classes
chat_prompt_explicit = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(
        "You are an expert {role}. Always respond in {style} tone."
    ),
    HumanMessagePromptTemplate.from_template("{question}")
])

# Format as messages
messages = chat_prompt.format_messages(
    domain="Python programming",
    user_input="How do I read a file?"
)
print(f"\nFormatted messages:")
for msg in messages:
    print(f"  {msg.type}: {msg.content[:50]}...")

# =============================================================================
# 3. MULTI-TURN CHAT TEMPLATE
# =============================================================================
print("\n" + "=" * 70)
print("3. MULTI-TURN CHAT TEMPLATE")
print("=" * 70)
print("""
USE CASE: Conversations with history, chatbots with memory
- Use MessagesPlaceholder for dynamic message history
- Maintains context across turns
""")

# Template with message history placeholder
chat_with_history = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Be concise."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{user_input}")
])

# Simulate conversation history
history = [
    HumanMessage(content="What's the capital of France?"),
    AIMessage(content="The capital of France is Paris."),
    HumanMessage(content="What's its population?"),
    AIMessage(content="Paris has about 2.1 million people in the city proper.")
]

# Format with history
messages_with_history = chat_with_history.format_messages(
    chat_history=history,
    user_input="What are some famous landmarks there?"
)
print(f"\nMessages with history ({len(messages_with_history)} total):")
for msg in messages_with_history:
    print(f"  {msg.type}: {msg.content[:40]}...")

# =============================================================================
# 4. PARTIAL TEMPLATES
# =============================================================================
print("\n" + "=" * 70)
print("4. PARTIAL TEMPLATES")
print("=" * 70)
print("""
USE CASE: Pre-fill some variables, create specialized variants
- Fix certain variables ahead of time
- Create template "factories"
""")

# Base template
base_template = PromptTemplate.from_template(
    "You are a {role}. Answer this {difficulty} question: {question}"
)

# Create partial with fixed role
teacher_template = base_template.partial(role="patient teacher")
expert_template = base_template.partial(role="technical expert")

# Use the partial templates
teacher_prompt = teacher_template.format(
    difficulty="beginner",
    question="What is a variable?"
)
print(f"\nTeacher prompt:\n{teacher_prompt}")

expert_prompt = expert_template.format(
    difficulty="advanced",
    question="Explain memory management in Python"
)
print(f"\nExpert prompt:\n{expert_prompt}")

# Partial with functions (dynamic values)
from datetime import datetime

def get_current_date():
    return datetime.now().strftime("%Y-%m-%d")

dynamic_template = PromptTemplate(
    template="Today is {date}. {query}",
    input_variables=["query"],
    partial_variables={"date": get_current_date}
)

print(f"\nDynamic template: {dynamic_template.format(query='What day is it?')}")

# =============================================================================
# 5. TEMPLATE COMPOSITION
# =============================================================================
print("\n" + "=" * 70)
print("5. TEMPLATE COMPOSITION")
print("=" * 70)
print("""
USE CASE: Build complex prompts from reusable parts
- Combine multiple templates
- Create modular prompt systems
""")

# Define reusable components
context_template = PromptTemplate.from_template(
    "Context: {context}\n\n"
)

instruction_template = PromptTemplate.from_template(
    "Instructions: {instructions}\n\n"
)

question_template = PromptTemplate.from_template(
    "Question: {question}\n\nAnswer:"
)

# Compose them together
from langchain_core.prompts import PipelinePromptTemplate

full_template = PromptTemplate.from_template(
    "{context}{instructions}{question}"
)

pipeline_prompt = PipelinePromptTemplate(
    final_prompt=full_template,
    pipeline_prompts=[
        ("context", context_template),
        ("instructions", instruction_template),
        ("question", question_template)
    ]
)

composed = pipeline_prompt.format(
    context="You are analyzing a sales report.",
    instructions="Be specific and use numbers.",
    question="What were the top 3 products?"
)
print(f"\nComposed prompt:\n{composed}")

# =============================================================================
# 6. PROMPT TEMPLATE WITH OUTPUT PARSER
# =============================================================================
print("\n" + "=" * 70)
print("6. PROMPT WITH FORMAT INSTRUCTIONS")
print("=" * 70)
print("""
USE CASE: Getting structured output from LLMs
- Include format instructions in prompt
- Guide LLM to produce parseable output
""")

from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import List

# Define output structure
class MovieReview(BaseModel):
    title: str = Field(description="The movie title")
    rating: int = Field(description="Rating from 1-10")
    pros: List[str] = Field(description="List of positive points")
    cons: List[str] = Field(description="List of negative points")

# Create parser
parser = PydanticOutputParser(pydantic_object=MovieReview)

# Create prompt with format instructions
review_prompt = PromptTemplate(
    template="""Analyze the following movie and provide a structured review.

Movie: {movie_name}

{format_instructions}""",
    input_variables=["movie_name"],
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

formatted_review = review_prompt.format(movie_name="Inception")
print(f"\nReview prompt with format instructions:\n{formatted_review[:500]}...")

# =============================================================================
# 7. CONDITIONAL PROMPT SECTIONS
# =============================================================================
print("\n" + "=" * 70)
print("7. CONDITIONAL PROMPT SECTIONS")
print("=" * 70)
print("""
USE CASE: Dynamic prompts that adapt based on input
- Include/exclude sections based on conditions
- Create flexible prompts
""")

def create_dynamic_prompt(include_examples: bool, include_constraints: bool):
    """Create a prompt with optional sections."""

    sections = ["You are a helpful coding assistant.\n"]

    if include_examples:
        sections.append("""
Examples of good responses:
- Explain the concept first
- Show code with comments
- Provide alternatives when relevant
""")

    if include_constraints:
        sections.append("""
Constraints:
- Keep responses under 200 words
- Use Python 3.10+ syntax
- Include type hints
""")

    sections.append("\nUser question: {question}\n\nResponse:")

    return PromptTemplate.from_template("".join(sections))

# Create different variants
simple_prompt = create_dynamic_prompt(False, False)
guided_prompt = create_dynamic_prompt(True, True)

print(f"\nSimple prompt:\n{simple_prompt.format(question='How do I sort a list?')}")
print(f"\nGuided prompt:\n{guided_prompt.format(question='How do I sort a list?')}")

# =============================================================================
# 8. EXAMPLE: DIFFERENT PROMPTS FOR DIFFERENT TASKS
# =============================================================================
print("\n" + "=" * 70)
print("8. TASK-SPECIFIC PROMPT EXAMPLES")
print("=" * 70)

# --- Summarization Prompt ---
summarization_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert summarizer. Create concise summaries that:
- Capture key points
- Maintain accuracy
- Use clear language
- Stay under {max_words} words"""),
    ("human", "Summarize this text:\n\n{text}")
])

# --- Code Review Prompt ---
code_review_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior code reviewer. Review code for:
- Bugs and errors
- Performance issues
- Security vulnerabilities
- Code style and best practices
- Potential improvements

Be constructive and specific."""),
    ("human", "Review this {language} code:\n\n```{language}\n{code}\n```")
])

# --- Creative Writing Prompt ---
creative_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a creative writing assistant.
Style: {style}
Tone: {tone}
Target audience: {audience}"""),
    ("human", "Write about: {topic}")
])

# --- Data Analysis Prompt ---
analysis_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data analyst. When analyzing data:
- Look for patterns and trends
- Identify outliers
- Provide statistical insights
- Make actionable recommendations"""),
    ("human", "Analyze this data:\n{data}\n\nFocus on: {focus_area}")
])

# --- Translation Prompt ---
translation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional translator.
Source language: {source_lang}
Target language: {target_lang}
Style: {style}

Maintain the original meaning, tone, and cultural nuances."""),
    ("human", "{text}")
])

print("Task-specific prompts created:")
print("  - summarization_prompt: For text summarization")
print("  - code_review_prompt: For reviewing code")
print("  - creative_prompt: For creative writing")
print("  - analysis_prompt: For data analysis")
print("  - translation_prompt: For language translation")

# =============================================================================
# 9. USING PROMPTS WITH CHAINS
# =============================================================================
print("\n" + "=" * 70)
print("9. USING PROMPTS IN CHAINS")
print("=" * 70)

model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
parser = StrOutputParser()

# Simple chain
simple_chain = summarization_prompt | model | parser

# Invoke (example - would need actual API call)
print("\nChain structure:")
simple_chain.get_graph().print_ascii()

# =============================================================================
# BEST PRACTICES
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("PROMPT TEMPLATE BEST PRACTICES")
    print("=" * 70)
    print("""
    1. USE CHAT TEMPLATES for chat models (GPT-4, Claude)
       - They understand roles (system, human, assistant)
       - Better results than plain text

    2. SYSTEM MESSAGE is crucial
       - Define persona, constraints, format
       - Set the tone for the entire conversation

    3. BE SPECIFIC in instructions
       - "Respond in JSON" vs "Respond in valid JSON with keys: name, age"
       - Ambiguity leads to inconsistent outputs

    4. USE EXAMPLES (few-shot) for complex tasks
       - Show the exact format you want
       - 2-3 examples usually sufficient

    5. INCLUDE FORMAT INSTRUCTIONS for structured output
       - Use PydanticOutputParser for type-safe outputs
       - Always validate LLM responses

    6. KEEP PROMPTS MODULAR
       - Use partial templates for variants
       - Compose complex prompts from parts

    7. VERSION YOUR PROMPTS
       - Prompts are code, treat them as such
       - Track changes, test improvements

    8. TEST EDGE CASES
       - Empty inputs, long inputs, special characters
       - Different languages if applicable
    """)
