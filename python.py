import getpass
import os
from langchain_core.messages import HumanMessage, SystemMessage

if not os.environ.get("OPENAI_API_KEY"):
  os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-4o-mini", model_provider="openai")

messages = [
    SystemMessage("Translate the following from English into Hindi"),
    HumanMessage("hi!"),
]

response = model.invoke(messages)
print(response.content)