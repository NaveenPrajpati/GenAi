from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.schema.runnable import RunnableParallel,RunnableSequence

load_dotenv()

prompt1=PromptTemplate(
    template='write a joke about {topic}',
    input_variables=['topic']
)
prompt2=PromptTemplate(
    template='generate a linkedin post about {topic}',
    input_variables=['topic']
)

model=ChatOpenAI()
parser=StrOutputParser()
chain=RunnableParallel({
    'joke':RunnableSequence(prompt1,model,parser),
    'linkedin':RunnableSequence(prompt2 ,model,parser)
})

print(chain.invoke({'topic':'brain'}))