# app.py
from __future__ import annotations

import os
import uuid
from typing import Annotated, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import ChatOpenAI

# ---------- Setup ----------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or st.secrets.get("OPENAI_API_KEY", "")

st.set_page_config(
    page_title="LangGraph Chat",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("💬 LangGraph Multi-Thread Chat")

if not OPENAI_API_KEY:
    st.error(
        "Missing OpenAI API key. Set OPENAI_API_KEY in your environment or Streamlit secrets."
    )
    st.stop()


# ---------- LangGraph State ----------
class ChatState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


@st.cache_resource(show_spinner=False)
def build_chatbot(api_key: str, model: str, temperature: float):
    """Create and compile the LangGraph with a memory checkpointer."""
    graph = StateGraph(ChatState)

    try:
        llm = ChatOpenAI(
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=30,  # Add timeout
            max_retries=2,  # Add retry logic
        )
    except Exception as e:
        st.error(f"Failed to initialize OpenAI client: {e}")
        st.stop()

    def chatNode(state: ChatState):
        messages = state["messages"]
        try:
            response = llm.invoke(messages)
            return {"messages": [response]}
        except Exception as e:
            error_msg = AIMessage(content=f"Error generating response: {str(e)}")
            return {"messages": [error_msg]}

    graph.add_node("chatNode", chatNode)
    graph.add_edge(START, "chatNode")
    graph.add_edge("chatNode", END)

    memory = InMemorySaver()
    chatbot = graph.compile(checkpointer=memory)
    return chatbot, memory


# ---------- Utility Functions ----------
def get_thread_display_name(thread_id: str, ui_messages: List[Dict[str, str]]) -> str:
    """Generate a friendly display name for a thread based on its first message."""
    if not ui_messages:
        return f"New Chat ({thread_id[:8]})"

    first_user_msg = next((msg for msg in ui_messages if msg["role"] == "user"), None)
    if first_user_msg:
        preview = first_user_msg["content"][:30]
        if len(first_user_msg["content"]) > 30:
            preview += "..."
        return preview
    return f"Chat ({thread_id[:8]})"


def clear_current_thread():
    """Clear all messages in the current thread."""
    current_thread = st.session_state.selected_thread
    st.session_state.ui_by_thread[current_thread] = []


# ---------- Session bootstrap ----------
def ensure_session_defaults():
    """Initialize session state with default values."""
    defaults = {
        "model": "gpt-4o-mini",
        "temperature": 0.7,
        "system_prompt": "You are a helpful AI assistant.",
        "thread_ids": [str(uuid.uuid4())],
        "ui_by_thread": {},
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

    # Set selected thread if not exists
    if "selected_thread" not in st.session_state:
        st.session_state.selected_thread = st.session_state.thread_ids[0]

    # Build chatbot if not exists
    if "chatbot_bundle" not in st.session_state:
        st.session_state.chatbot_bundle = build_chatbot(
            OPENAI_API_KEY, st.session_state.model, st.session_state.temperature
        )

    # Ensure current thread has a message list
    st.session_state.ui_by_thread.setdefault(st.session_state.selected_thread, [])


def switch_thread(thread_id: str):
    """Switch to a different thread."""
    if thread_id in st.session_state.thread_ids:
        st.session_state.selected_thread = thread_id
        st.session_state.ui_by_thread.setdefault(thread_id, [])


def new_thread():
    """Create a new chat thread."""
    tid = str(uuid.uuid4())
    st.session_state.thread_ids.append(tid)
    switch_thread(tid)


def delete_thread(thread_id: str):
    """Delete a thread and switch to another or create new one."""
    if len(st.session_state.thread_ids) <= 1:
        # Don't delete the last thread, just clear it
        clear_current_thread()
        return

    # Remove from lists
    st.session_state.ui_by_thread.pop(thread_id, None)
    st.session_state.thread_ids = [
        t for t in st.session_state.thread_ids if t != thread_id
    ]

    # Switch to another thread
    if st.session_state.selected_thread == thread_id:
        switch_thread(st.session_state.thread_ids[-1])


ensure_session_defaults()

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🔧 Controls")

    # Model settings
    with st.expander("Model Settings", expanded=False):
        available_models = [
            "gpt-4o",
            "gpt-4o-mini",
            "gpt-4-turbo",
            "gpt-3.5-turbo",
        ]

        model = st.selectbox(
            "Model",
            available_models,
            index=(
                available_models.index(st.session_state.model)
                if st.session_state.model in available_models
                else 1
            ),
            help="Select the OpenAI chat model to use.",
        )

        temperature = st.slider(
            "Temperature",
            0.0,
            1.0,
            st.session_state.temperature,
            0.05,
            help="Controls randomness: 0 = deterministic, 1 = very creative",
        )

        sys_prompt = st.text_area(
            "System Prompt",
            value=st.session_state.system_prompt,
            placeholder="You are a helpful assistant...",
            height=100,
            help="Instructions that define the AI's behavior and role",
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Apply Changes", use_container_width=True):
                st.session_state.model = model
                st.session_state.temperature = float(temperature)
                st.session_state.system_prompt = sys_prompt
                # Rebuild chatbot with new settings
                st.session_state.chatbot_bundle = build_chatbot(
                    OPENAI_API_KEY, st.session_state.model, st.session_state.temperature
                )
                st.success("✅ Settings updated!")
                st.rerun()

        with col2:
            if st.button("Reset to Default", use_container_width=True):
                st.session_state.model = "gpt-4o-mini"
                st.session_state.temperature = 0.7
                st.session_state.system_prompt = "You are a helpful AI assistant."
                st.rerun()

    st.divider()

    # Thread management
    st.subheader("💬 Chat Threads")

    # Display threads with friendly names
    thread_options = []
    thread_labels = []
    for tid in st.session_state.thread_ids:
        ui_msgs = st.session_state.ui_by_thread.get(tid, [])
        display_name = get_thread_display_name(tid, ui_msgs)
        thread_options.append(tid)
        thread_labels.append(display_name)

    if thread_options:
        selected_index = (
            thread_options.index(st.session_state.selected_thread)
            if st.session_state.selected_thread in thread_options
            else 0
        )

        selected_thread = st.selectbox(
            "Select Thread",
            thread_options,
            index=selected_index,
            format_func=lambda x: thread_labels[thread_options.index(x)],
            label_visibility="collapsed",
        )

        if selected_thread != st.session_state.selected_thread:
            switch_thread(selected_thread)
            st.rerun()

    # Thread action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🆕 New", use_container_width=True, help="Create new thread"):
            new_thread()
            st.rerun()

    with col2:
        if st.button(
            "🗑️ Delete",
            use_container_width=True,
            help="Delete current thread",
            disabled=len(st.session_state.thread_ids) <= 1,
        ):
            delete_thread(st.session_state.selected_thread)
            st.rerun()

    # Additional thread actions
    col3, col4 = st.columns(2)
    with col3:
        if st.button("🧹 Clear", use_container_width=True, help="Clear current thread"):
            clear_current_thread()
            st.rerun()

    with col4:
        total_msgs = sum(len(msgs) for msgs in st.session_state.ui_by_thread.values())
        if st.button(f"📊 Stats", use_container_width=True):
            st.info(
                f"**Threads:** {len(st.session_state.thread_ids)}\n**Total Messages:** {total_msgs}"
            )

    st.caption("💡 Each thread maintains independent conversation history")

# ---------- Main chat area ----------
chatbot, memory = st.session_state.chatbot_bundle
thread_id = st.session_state.selected_thread
ui_messages = st.session_state.ui_by_thread[thread_id]

# Display current thread info
current_thread_name = get_thread_display_name(thread_id, ui_messages)
st.caption(
    f"**Current Thread:** {current_thread_name} • **Messages:** {len(ui_messages)}"
)

# Render conversation history
if not ui_messages:
    st.info("👋 Start a new conversation by typing a message below!")

for i, message in enumerate(ui_messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # Add copy button for long messages
        if len(message["content"]) > 100:
            if st.button(f"📋 Copy", key=f"copy_{thread_id}_{i}", help="Copy message"):
                st.code(message["content"])

# Chat input
prompt = st.chat_input("Type your message here...", key=f"input_{thread_id}")

if prompt:
    # Display user message
    ui_messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Prepare messages for the model
    input_messages: List[BaseMessage] = []

    # Add system message if provided
    if st.session_state.system_prompt.strip():
        input_messages.append(SystemMessage(content=st.session_state.system_prompt))

    # Add conversation history from memory (for context)
    config = {"configurable": {"thread_id": thread_id}}
    try:
        # Get existing conversation state
        current_state = chatbot.get_state(config)
        if current_state and current_state.values.get("messages"):
            # Add existing messages for context
            input_messages.extend(current_state.values["messages"])
    except Exception:
        # If no prior state, start fresh
        pass

    # Add current user message
    input_messages.append(HumanMessage(content=prompt))

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = chatbot.invoke({"messages": input_messages}, config=config)

                if result and result.get("messages"):
                    ai_message = result["messages"][-1]
                    ai_text = (
                        ai_message.content
                        if hasattr(ai_message, "content")
                        else str(ai_message)
                    )
                else:
                    ai_text = (
                        "I'm sorry, I couldn't generate a response. Please try again."
                    )

            except Exception as e:
                ai_text = (
                    f"⚠️ **Error:** {str(e)}\n\nPlease check your API key and try again."
                )
                st.error("Failed to generate response")

        # Display and store AI response
        st.markdown(ai_text)
        ui_messages.append({"role": "assistant", "content": ai_text})

# Auto-scroll to bottom
if ui_messages:
    st.markdown(
        """
        <script>
        var element = window.parent.document.querySelector('.main .block-container');
        element.scrollTop = element.scrollHeight;
        </script>
        """,
        unsafe_allow_html=True,
    )
