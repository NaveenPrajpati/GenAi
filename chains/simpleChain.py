from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

prompt= PromptTemplate(
    template='Generate 5 intersting facts about {topic}',
    input_variables=['topic']
)

model=init_chat_model('gpt-4.1-nano',model_provider='openai')

parser= StrOutputParser()

chain =prompt | model | parser

result=chain.invoke({'topic':'plants'})
print(result)

chain.get_graph().print_ascii()