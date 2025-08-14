# app.py
from __future__ import annotations

import os
import uuid
from typing import Annotated

import streamlit as st
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

# ---------- Setup ----------
load_dotenv()  # loads .env if present (e.g., OPENAI_API_KEY=...)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

st.set_page_config(page_title="LangGraph Chat", page_icon="💬", layout="centered")
st.title("💬 LangGraph Chat ")

if not OPENAI_API_KEY:
    st.error(
        "Missing OpenAI API key. Set OPENAI_API_KEY in your environment or Streamlit secrets."
    )
    st.stop()


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


def build_chatbot() -> tuple:
    """Create and compile the LangGraph with a memory checkpointer."""
    graph = StateGraph(ChatState)

    llm = ChatOpenAI(api_key=OPENAI_API_KEY)

    def chatNode(state: ChatState):
        messages = state["messages"]
        response = llm.invoke(messages)
        return {"messages": [response]}

    graph.add_node("chatNode", chatNode)
    graph.add_edge(START, "chatNode")
    graph.add_edge("chatNode", END)

    memory = InMemorySaver()
    chatbot = graph.compile(checkpointer=memory)
    return chatbot, memory


# ---------- Session Bootstrap ----------
if "chatbot" not in st.session_state:
    st.session_state.chatbot, st.session_state.memory = build_chatbot()

if "thread_id" not in st.session_state:
    st.session_state.thread_id = []
    st.session_state.thread_id.append(uuid.uuid4())

if "ui_messages" not in st.session_state:
    # For Streamlit display; LangGraph keeps its own message history by thread_id
    st.session_state.ui_messages: list[dict] = []

# ---------- Sidebar ----------
with st.sidebar:
    st.header("Controls")
    st.caption("Each chat uses a distinct LangGraph thread with checkpointed memory.")

    if st.button("🆕 New chat", use_container_width=True):
        st.session_state.thread_id.append(uuid.uuid4())
        st.session_state.ui_messages = []
        st.rerun()

    st.write("**Thread ID**")
    for x in st.session_state.thread_id:
        st.code(x)


# ---------- Render prior messages ----------
for m in st.session_state.ui_messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Chat input ----------
prompt = st.chat_input("Type your question…")

if prompt:
    # Show user message
    st.session_state.ui_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Invoke LangGraph with the current thread_id
    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    result = st.session_state.chatbot.invoke(
        {"messages": [HumanMessage(content=prompt)]}, config=config
    )

    # Extract assistant response
    ai_text = (
        result["messages"][-1].content if result and result.get("messages") else ""
    )
    if not ai_text:
        ai_text = "_(No response)_"

    st.session_state.ui_messages.append({"role": "assistant", "content": ai_text})
    with st.chat_message("assistant"):
        st.markdown(ai_text)
