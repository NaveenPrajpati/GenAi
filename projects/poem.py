from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, FewShotPromptTemplate
from langchain.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

examples = [
    {
        "topic": "sunrise",
        "mood": "cheerful",
        "poem": "Golden rays stretch wide,\nPainting clouds with joy and light,\nMorning's gift unfolds.",
    },
    {
        "topic": "rain",
        "mood": "melancholic",
        "poem": "Silver drops whisper\nSecrets to the thirsty earth,\nTears of clouded sky.",
    },
    {
        "topic": "ocean",
        "mood": "peaceful",
        "poem": "Waves embrace the shore,\nEndless rhythm soothes the soul,\nBlue serenity.",
    },
    {
        "topic": "city",
        "mood": "energetic",
        "poem": "Neon lights pulse bright,\nStreets alive with dreams and rush,\nUrban symphony.",
    },
]

# Create the example template
example_template = PromptTemplate.from_template(
    "Topic: {topic}\nMood: {mood}\nPoem:\n{poem}",
)

# Create the few-shot prompt template
few_shot_prompt = FewShotPromptTemplate(
    examples=examples,
    example_prompt=example_template,
    prefix="You are a skilled poet. Write haiku poems (3 lines, 5-7-5 syllable pattern) that capture the essence and mood of the given topic.\n\nHere are some examples:",
    suffix="\nNow write a haiku poem:\nTopic: {topic}\nMood: {mood}\nPoem:",
    input_variables=["topic", "mood"],
)

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-3.5-turbo",  # Specify model explicitly
    temperature=0.7,  # Add some creativity
    max_tokens=150,  # Limit response length
)

# Create output parser
parser = StrOutputParser()

# Create the chain using LCEL (LangChain Expression Language)
chain = few_shot_prompt | llm | parser


# Test the chain
def generate_poem(topic: str, mood: str) -> str:
    """Generate a poem for the given topic and mood"""
    try:
        response = chain.invoke({"topic": topic, "mood": mood})
        return response.strip()
    except Exception as e:
        return f"Error generating poem: {str(e)}"


test_cases = [
    ("night sky", "wondering"),
    ("autumn leaves", "nostalgic"),
    ("morning coffee", "cozy"),
    ("thunderstorm", "dramatic"),
]

# poem = generate_poem(topic, mood)
# print(poem)


# Interactive version
def interactive_poem_generator():
    """Interactive poem generation"""
    print("\n🎨 Interactive Poem Generator")
    print("Type 'quit' to exit")

    while True:
        print("\n" + "=" * 40)
        topic = input("Enter a topic: ").strip()

        if topic.lower() == "quit":
            print("Goodbye! 👋")
            break

        mood = input("Enter a mood: ").strip()

        if not topic or not mood:
            print("Please enter both topic and mood.")
            continue

        print(f"\nGenerating poem about '{topic}' with '{mood}' mood...")
        print("-" * 40)

        poem = generate_poem(topic, mood)
        print(poem)


# Advanced version with multiple styles
class AdvancedPoemGenerator:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.8, max_tokens=200)
        self.parser = StrOutputParser()

        # Different examples for different styles
        self.style_examples = {
            "haiku": [
                {
                    "topic": "cherry blossom",
                    "mood": "serene",
                    "poem": "Petals drift gently,\nSpring's whisper on morning breeze,\nBeauty in farewell.",
                },
                {
                    "topic": "mountain",
                    "mood": "majestic",
                    "poem": "Stone peaks touch the sky,\nSilent guardians of time,\nEternal and proud.",
                },
            ],
            "free_verse": [
                {
                    "topic": "memory",
                    "mood": "nostalgic",
                    "poem": "Photographs yellow with age\nhold moments\nthat once felt infinite\n\nNow they're fragments\nof a life\nlived in sepia tones",
                },
                {
                    "topic": "love",
                    "mood": "passionate",
                    "poem": "Your laughter\nspills like wine\nacross my heart\n\nStaining everything\nwith the color\nof possibility",
                },
            ],
            "limerick": [
                {
                    "topic": "cat",
                    "mood": "humorous",
                    "poem": "There once was a cat from Peru,\nWho dreamed of a mouse or two,\nHe'd chase his own tail,\nWithout any fail,\nThen nap when his dreaming was through!",
                }
            ],
        }

    def create_style_prompt(self, style: str):
        """Create a prompt template for a specific style"""
        examples = self.style_examples.get(style, self.style_examples["haiku"])

        style_instructions = {
            "haiku": "Write a haiku (3 lines, 5-7-5 syllable pattern)",
            "free_verse": "Write a free verse poem with natural rhythm and vivid imagery",
            "limerick": "Write a humorous limerick (5 lines, AABBA rhyme scheme)",
        }

        example_template = PromptTemplate(
            input_variables=["topic", "mood", "poem"],
            template="Topic: {topic}\nMood: {mood}\nPoem:\n{poem}",
        )

        return FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_template,
            prefix=f"You are a skilled poet. {style_instructions.get(style, 'Write a poem')} that captures the essence and mood of the given topic.\n\nHere are some examples:",
            suffix=f"\nNow write a {style} poem:\nTopic: {{topic}}\nMood: {{mood}}\nPoem:",
            input_variables=["topic", "mood"],
        )

    def generate_poem(self, topic: str, mood: str, style: str = "haiku") -> str:
        """Generate a poem with specified style"""
        try:
            prompt = self.create_style_prompt(style)
            chain = prompt | self.llm | self.parser

            response = chain.invoke({"topic": topic, "mood": mood})
            return response.strip()
        except Exception as e:
            return f"Error generating {style} poem: {str(e)}"


# Example usage of advanced generator
def demo_advanced_generator():
    generator = AdvancedPoemGenerator()

    styles = ["haiku", "free_verse", "limerick"]
    topic = "coffee shop"
    mood = "cozy"

    for style in styles:
        print(f"\n{style.upper()} - Topic: {topic}, Mood: {mood}")
        print("-" * 40)
        poem = generator.generate_poem(topic, mood, style)
        print(poem)


if __name__ == "__main__":
    # Run basic examples
    print("Basic Examples:")
    # (basic examples will run automatically)

    # Run advanced demo
    demo_advanced_generator()

    # Uncomment to run interactive version
    # interactive_poem_generator()
