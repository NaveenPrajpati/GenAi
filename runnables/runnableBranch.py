from langchain.schema.runnable import RunnableLambda, RunnableBranch, RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

# Base LLM and parser
llm = ChatOpenAI()
parser = StrOutputParser()

# Prompt templates
animal_prompt = PromptTemplate.from_template("Tell me a funny animal joke.")
tech_prompt = PromptTemplate.from_template("Tell me a tech joke.")
default_prompt = PromptTemplate.from_template("Tell me a joke about {topic}.")

# Chains
animal_chain = RunnableSequence(animal_prompt, llm, parser)
tech_chain = RunnableSequence(tech_prompt, llm, parser)
default_chain = RunnableSequence(default_prompt, llm, parser)

# Condition checkers
def is_animal(input: dict) -> bool:
    return input.get("topic", "").lower() == "animal"

def is_tech(input: dict) -> bool:
    return input.get("topic", "").lower() == "technology"

# Branching chain
branch_chain = RunnableBranch(
    (is_animal, animal_chain),
    (is_tech, tech_chain),
    default_chain  # default fallback
)

# Run it
if __name__ == "__main__":
    inputs = [
        {"topic": "animal"},
        {"topic": "technology"},
        {"topic": "cooking"}
    ]

    for i in inputs:
        print(f"Input: {i['topic']}")
        print("Joke:", branch_chain.invoke(i))
        print("-" * 40)