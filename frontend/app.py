import os

import requests
import streamlit as st

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖", layout="wide")


def _backend_url() -> str:
    # Try secrets if present, otherwise env var, then default
    try:
        return st.secrets["BACKEND_URL"]  # may raise FileNotFoundError if no secrets.toml
    except Exception:
        return os.environ.get("BACKEND_URL", "http://backend:8000")


BACKEND = _backend_url()


def stream_chat(messages, use_rag=True, k=3):
    url = f"{BACKEND}/chat/stream"
    payload = {"messages": messages, "use_rag": use_rag, "k": k}
    with requests.post(url, json=payload, stream=True, timeout=300) as r:
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")  # keep as-is (no .strip())
            if line.startswith("data:"):
                # Keep the model’s leading spaces in the token!
                # SSE line format is "data: <chunk>". We only drop the one space after "data:".
                chunk = line[len("data:") :]
                if chunk.startswith(" "):
                    chunk = chunk[1:]
                yield chunk


st.title("RAG Chatbot")

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    use_rag = st.toggle("Use RAG", value=True)
    k = st.slider("Top-K", 1, 8, 3)

# Keep full chat history
if "chat" not in st.session_state:
    st.session_state.chat = []

# Render history
for m in st.session_state.chat:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# Input & streaming reply
if prompt := st.chat_input("Ask me anything..."):
    st.session_state.chat.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        slot = st.empty()
        buf = []
        for chunk in stream_chat(st.session_state.chat, use_rag=use_rag, k=k):
            buf.append(chunk)
            slot.markdown("".join(buf))
        final = "".join(buf)

    st.session_state.chat.append({"role": "assistant", "content": final})
