from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain.schema.runnable import RunnableParallel
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Function that counts words in a joke (a string)
def count_words(joke: str) -> int:
    return len(joke.split())

# Prompt to generate a joke
joke_prompt = PromptTemplate(
    input_variables=["topic"],
    template="Write a joke about {topic}"
)

# LLM + parser
llm = ChatOpenAI()
parser = StrOutputParser()

# Step 1: Generate a joke
joke_chain = RunnableSequence(joke_prompt, llm, parser)

# Step 2: Count words in that joke
word_count_chain = joke_chain | RunnableLambda(count_words)

# Combine both: return both joke and word count
combined_chain = RunnableParallel({
    "joke": joke_chain,
    "wordCount": word_count_chain
})

# Run the chain
if __name__ == "__main__":
    input_data = {"topic": "keyboard and mouse"}
    result = combined_chain.invoke(input_data)
    print(result)