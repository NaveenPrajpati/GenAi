"""
Chat Prompt Templates - Advanced Patterns
==========================================

LEARNING OBJECTIVES:
- Master ChatPromptTemplate for modern LLMs
- Understand message roles and their purposes
- Build conversation-aware prompts
- Handle multi-turn dialogues

CONCEPT:
Chat models (GPT-4, Claude, Gemini) are designed to work with
structured messages that have roles. ChatPromptTemplate helps
you create these structured prompts.

MESSAGE ROLES:
    ┌─────────────────────────────────────────────────────────────┐
    │                     Message Role Types                       │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  SYSTEM:                                                    │
    │  - Sets the AI's persona, capabilities, constraints         │
    │  - Always first in the conversation                         │
    │  - Example: "You are a helpful Python tutor..."             │
    │                                                              │
    │  HUMAN (user):                                              │
    │  - The user's input/question                                │
    │  - What the AI should respond to                            │
    │  - Example: "How do I sort a list?"                         │
    │                                                              │
    │  AI (assistant):                                            │
    │  - Previous AI responses (for context)                      │
    │  - Used in few-shot examples                                │
    │  - Example: "You can use sorted() or .sort()..."            │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘

CONVERSATION FLOW:
    System → Human → AI → Human → AI → Human → AI ...
       ↑       ↑      ↑      ↑
    (persona) (Q1)  (A1)   (Q2)  ...

PREREQUISITES:
- Completed: prompts/promptTemplates.py

NEXT STEPS:
- prompts/fewShotPrompts.py - Examples-based prompting
- langgraph/basicChatbot.py - Chatbot with memory
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
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
# 1. BASIC CHAT PROMPT
# =============================================================================
print("=" * 70)
print("1. BASIC CHAT PROMPT")
print("=" * 70)
print("""
USE CASE: Simple chat interactions with persona
""")

# Using tuple syntax (shorthand)
basic_chat = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly assistant who speaks like a pirate."),
    ("human", "{question}")
])

messages = basic_chat.format_messages(question="What's the weather like?")
print("\nFormatted messages:")
for m in messages:
    print(f"  [{m.type}]: {m.content}")

# =============================================================================
# 2. DETAILED SYSTEM PROMPTS
# =============================================================================
print("\n" + "=" * 70)
print("2. DETAILED SYSTEM PROMPTS")
print("=" * 70)
print("""
USE CASE: Define complex AI behavior, constraints, and output format
""")

# Detailed system prompt for a specific task
detailed_system = """You are an expert technical writer.

ROLE: Create clear, concise documentation.

STYLE GUIDELINES:
- Use active voice
- Keep sentences short (max 20 words)
- Use bullet points for lists
- Include code examples when relevant

OUTPUT FORMAT:
- Start with a brief overview (1-2 sentences)
- Use headers for sections
- End with a "See Also" section

CONSTRAINTS:
- Stay factual, no speculation
- Cite sources when possible
- Use {language} programming language for examples"""

documentation_prompt = ChatPromptTemplate.from_messages([
    ("system", detailed_system),
    ("human", "Document the following concept: {topic}")
])

# Format with variables
formatted = documentation_prompt.format_messages(
    language="Python",
    topic="list comprehensions"
)
print(f"\nSystem message length: {len(formatted[0].content)} characters")
print(f"Preview: {formatted[0].content[:200]}...")

# =============================================================================
# 3. ROLE-BASED PROMPTS
# =============================================================================
print("\n" + "=" * 70)
print("3. ROLE-BASED PROMPTS")
print("=" * 70)
print("""
USE CASE: Different personas for different tasks
""")

# Define different roles
ROLES = {
    "teacher": """You are a patient teacher who:
- Explains concepts step by step
- Uses simple analogies
- Asks questions to check understanding
- Encourages the student""",

    "critic": """You are a critical reviewer who:
- Points out flaws and weaknesses
- Suggests improvements
- Is direct but constructive
- Backs up criticism with reasoning""",

    "optimist": """You are an optimistic advisor who:
- Focuses on possibilities
- Finds silver linings
- Encourages action
- Maintains positive energy""",

    "analyst": """You are a data analyst who:
- Focuses on facts and numbers
- Avoids emotional language
- Provides statistical insights
- Presents multiple interpretations"""
}


def create_role_prompt(role_name: str):
    """Create a chat prompt with a specific role."""
    if role_name not in ROLES:
        raise ValueError(f"Unknown role: {role_name}")

    return ChatPromptTemplate.from_messages([
        ("system", ROLES[role_name]),
        ("human", "{input}")
    ])


# Create prompts for different roles
teacher_prompt = create_role_prompt("teacher")
critic_prompt = create_role_prompt("critic")

print("\nTeacher prompt:")
print(teacher_prompt.format_messages(input="Explain recursion")[0].content[:100])

print("\nCritic prompt:")
print(critic_prompt.format_messages(input="Review my code")[0].content[:100])

# =============================================================================
# 4. CONVERSATION HISTORY
# =============================================================================
print("\n" + "=" * 70)
print("4. CONVERSATION HISTORY")
print("=" * 70)
print("""
USE CASE: Multi-turn conversations, chatbots with memory
""")

# Prompt with history placeholder
chat_with_memory = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant with perfect memory. "
               "Reference previous parts of the conversation when relevant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Simulate conversation history
conversation_history = [
    HumanMessage(content="My name is Alice"),
    AIMessage(content="Hello Alice! Nice to meet you. How can I help you today?"),
    HumanMessage(content="I'm learning Python"),
    AIMessage(content="That's great, Alice! Python is an excellent choice. "
                      "What aspect of Python would you like to focus on?"),
]

# New message with full context
messages = chat_with_memory.format_messages(
    history=conversation_history,
    input="What should I learn first?"
)

print(f"\nTotal messages in context: {len(messages)}")
print("Message flow:")
for m in messages:
    preview = m.content[:50] + "..." if len(m.content) > 50 else m.content
    print(f"  [{m.type}]: {preview}")

# =============================================================================
# 5. OPTIONAL PLACEHOLDERS
# =============================================================================
print("\n" + "=" * 70)
print("5. OPTIONAL PLACEHOLDERS")
print("=" * 70)
print("""
USE CASE: Prompts that work with or without certain context
""")

# History is optional
flexible_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a coding assistant."),
    MessagesPlaceholder(variable_name="history", optional=True),
    ("human", "{question}")
])

# Works without history
without_history = flexible_prompt.format_messages(
    question="How do I reverse a string?"
)
print(f"\nWithout history: {len(without_history)} messages")

# Works with history
with_history = flexible_prompt.format_messages(
    history=[
        HumanMessage(content="I'm using Python 3.10"),
        AIMessage(content="Great! Python 3.10 has many useful features.")
    ],
    question="How do I reverse a string?"
)
print(f"With history: {len(with_history)} messages")

# =============================================================================
# 6. MULTI-ASSISTANT CONVERSATIONS
# =============================================================================
print("\n" + "=" * 70)
print("6. MULTI-ASSISTANT CONVERSATIONS")
print("=" * 70)
print("""
USE CASE: Simulating multiple AI personas, debate scenarios
""")

debate_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are moderating a debate. Present balanced views."),
    ("human", "Topic: {topic}"),
    ("ai", "As the optimist, I believe {topic} will lead to positive outcomes because..."),
    ("human", "What does the pessimist think?"),
    ("ai", "As the pessimist, I'm concerned about {topic} because..."),
    ("human", "Now give me your balanced analysis.")
])

messages = debate_prompt.format_messages(topic="AI in education")
print("\nDebate prompt structure:")
for i, m in enumerate(messages):
    print(f"  {i+1}. [{m.type}]: {m.content[:40]}...")

# =============================================================================
# 7. CONTEXT INJECTION PATTERNS
# =============================================================================
print("\n" + "=" * 70)
print("7. CONTEXT INJECTION PATTERNS")
print("=" * 70)
print("""
USE CASE: RAG systems, providing relevant context
""")

# RAG-style prompt
rag_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions based on provided context.

INSTRUCTIONS:
- Only use information from the context below
- If the answer isn't in the context, say "I don't have that information"
- Quote relevant parts of the context when answering
- Be concise but complete"""),
    ("human", """CONTEXT:
{context}

QUESTION: {question}

ANSWER:""")
])

# Example usage
formatted = rag_prompt.format_messages(
    context="Python was created by Guido van Rossum in 1991. "
            "It emphasizes code readability and simplicity.",
    question="Who created Python?"
)
print("\nRAG prompt formatted:")
print(f"  Human message preview: {formatted[1].content[:100]}...")

# =============================================================================
# 8. OUTPUT FORMAT SPECIFICATION
# =============================================================================
print("\n" + "=" * 70)
print("8. OUTPUT FORMAT SPECIFICATION")
print("=" * 70)
print("""
USE CASE: Getting structured, parseable responses
""")

json_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a data extraction assistant.

ALWAYS respond with valid JSON in this exact format:
{{
    "extracted_entities": [
        {{"name": "entity name", "type": "person/place/org", "confidence": 0.0-1.0}}
    ],
    "summary": "brief summary",
    "sentiment": "positive/negative/neutral"
}}

Do not include any text outside the JSON."""),
    ("human", "Extract entities from: {text}")
])

markdown_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a documentation writer.

ALWAYS format responses in Markdown with:
- # Headers for main sections
- ## Subheaders for subsections
- `code` for inline code
- ```language for code blocks
- - bullet points for lists
- **bold** for emphasis"""),
    ("human", "Document: {topic}")
])

print("Format-specific prompts created:")
print("  - json_prompt: Forces JSON output")
print("  - markdown_prompt: Forces Markdown output")

# =============================================================================
# 9. CHAIN-OF-THOUGHT PROMPTS
# =============================================================================
print("\n" + "=" * 70)
print("9. CHAIN-OF-THOUGHT PROMPTS")
print("=" * 70)
print("""
USE CASE: Complex reasoning, step-by-step problem solving
""")

cot_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a logical problem solver.

For every problem:
1. First, understand what is being asked
2. Break down the problem into smaller parts
3. Solve each part step by step
4. Combine the solutions
5. Verify your answer

Always show your reasoning process before giving the final answer.
Format:
UNDERSTANDING: ...
STEPS: ...
SOLUTION: ...
VERIFICATION: ...
FINAL ANSWER: ..."""),
    ("human", "{problem}")
])

# =============================================================================
# 10. PERSONA SWITCHING
# =============================================================================
print("\n" + "=" * 70)
print("10. PERSONA SWITCHING")
print("=" * 70)
print("""
USE CASE: Dynamic persona based on user needs
""")

adaptive_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an adaptive assistant.

Based on the user's expertise level ({level}), adjust your response:

BEGINNER:
- Use simple language
- Avoid jargon
- Give examples
- Be encouraging

INTERMEDIATE:
- Use some technical terms
- Provide more depth
- Reference best practices

EXPERT:
- Use technical language freely
- Focus on edge cases
- Discuss trade-offs
- Be concise"""),
    ("human", "{question}")
])

# Same question, different levels
for level in ["beginner", "expert"]:
    formatted = adaptive_prompt.format_messages(
        level=level,
        question="How do decorators work?"
    )
    print(f"\n{level.upper()} system message preview:")
    print(f"  {formatted[0].content[100:200]}...")

# =============================================================================
# 11. PRACTICAL EXAMPLE: COMPLETE CHATBOT PROMPT
# =============================================================================
print("\n" + "=" * 70)
print("11. COMPLETE CHATBOT PROMPT")
print("=" * 70)

chatbot_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a customer support agent for TechCorp.

COMPANY INFO:
- Products: Software tools for developers
- Support hours: 24/7
- Refund policy: 30 days, no questions asked

YOUR CAPABILITIES:
- Answer product questions
- Help with technical issues
- Process refund requests
- Escalate complex issues to human agents

GUIDELINES:
- Be friendly and professional
- Use the customer's name when known
- Keep responses under 100 words unless technical detail needed
- If unsure, say "Let me check that for you" and ask to escalate

CURRENT CUSTOMER:
Name: {customer_name}
Subscription: {subscription_tier}
Account age: {account_age}"""),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("human", "{message}")
])

# Example usage
messages = chatbot_prompt.format_messages(
    customer_name="John",
    subscription_tier="Premium",
    account_age="2 years",
    chat_history=[],
    message="I need help with my subscription"
)

print("\nChatbot prompt ready with customer context:")
print(f"  System message: {len(messages[0].content)} characters")
print(f"  Customer: John (Premium, 2 years)")

# =============================================================================
# BEST PRACTICES
# =============================================================================
if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("CHAT PROMPT BEST PRACTICES")
    print("=" * 70)
    print("""
    1. SYSTEM MESSAGE STRUCTURE:
       - Who the AI is (persona)
       - What it can/cannot do (capabilities)
       - How to respond (format/style)
       - Relevant context (data, constraints)

    2. KEEP HISTORY RELEVANT:
       - Summarize old messages if too long
       - Use MessagesPlaceholder for flexibility
       - Consider token limits

    3. BE EXPLICIT ABOUT FORMAT:
       - Show exact output structure expected
       - Use examples in system message
       - Validate outputs

    4. USE APPROPRIATE ROLES:
       - System: Setup and rules
       - Human: User input
       - AI: Examples or previous responses

    5. TEST EDGE CASES:
       - Empty history
       - Very long inputs
       - Ambiguous questions
       - Off-topic requests
    """)
