from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate.from_template(
    """
You are a senior software engineer performing a code review.
Step through the following code in detail.
For each relevant part:
1. Explain what it does.
2. Identify potential bugs or issues.
3. Provide suggestions for improvement.
4. Summarize all findings concisely.
Code:
{code}
""",
)

llm = ChatOpenAI()
chain = prompt | llm

result = chain.invoke({"code": "def foo(x): return x+1/0"})
print(result)
