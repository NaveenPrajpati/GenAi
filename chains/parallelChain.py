from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.chat_models import init_chat_model
from langchain.schema.runnable import RunnableParallel

load_dotenv()

prompt1= PromptTemplate(
    template='Generate short and simple notes on topic {text}',
    input_variables=['text']
)
prompt2= PromptTemplate(
    template='Generate  5 short question and answer from following text  \n {text}',
    input_variables=['text']
)
prompt3= PromptTemplate(
    template='Merge the provide notes and quiz into a single document \n notes -> {notes} and quiz {quiz}',
    input_variables=['text']
)

model1=init_chat_model('gemini-2.0-flash',model_provider='google_genai')
model2=ChatOpenAI()
 
parser= StrOutputParser()

parallelChain=RunnableParallel({
    'notes':prompt1 | model1 | parser,
    'quiz': prompt2 | model2 | parser
})

mergeChain = prompt3 | model1 | parser

chain = parallelChain | mergeChain
result=chain.invoke({'text':'react.js'})
print(result)

chain.get_graph().print_ascii()