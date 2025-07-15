from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model

load_dotenv()

prompt1= PromptTemplate(
    template='Generate a detailed report on {topic}',
    input_variables=['topic']
)
prompt2= PromptTemplate(
    template='Generate a 5 point summary from the following text \n {text}',
    input_variables=['text']
)

model=init_chat_model('gpt-4.1-nano',model_provider='openai')
 
parser= StrOutputParser()

chain = prompt1 | model | parser| prompt2 | model | parser

result=chain.invoke({'topic':'gen ai'})
print(result)

chain.get_graph().print_ascii()