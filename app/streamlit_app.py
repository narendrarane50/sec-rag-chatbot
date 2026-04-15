import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import requests as http_requests
from rag.retrieval_chain import load_collection, retrieve, build_prompt

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="SEC Filing RAG Chatbot",
    page_icon="📈",
    layout="wide",
)

TICKERS = ["All", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "JPM", "BAC", "GS"]

# ── Cache the collection so it only loads once ────────────────
@st.cache_resource
def get_collection():
    return load_collection()

def query_ollama(prompt: str) -> str:
    try:
        resp = http_requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "repeat_penalty": 1.1,
                    "num_predict": 512,
                }
            },
            timeout=120,
        )
        return resp.json().get("response", "").strip()
    except Exception as e:
        return f"Error contacting Ollama: {e}. Make sure 'ollama serve' is running."

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.title("📈 SEC RAG Chatbot")
    st.markdown("Ask questions grounded in real SEC 10-K filings.")
    st.divider()

    ticker = st.selectbox("Filter by company", TICKERS)
    st.divider()

    st.markdown("**Example questions**")
    examples = [
        "What was total revenue?",
        "What were the main risk factors?",
        "What was net income?",
        "How did operating expenses change?",
        "What is the company's cash position?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["prefill"] = ex

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state["messages"] = []

# ── Main area ─────────────────────────────────────────────────
st.title("SEC 10-K Filing Analyst")
st.caption("Powered by Mistral + RAG over real SEC filings · 10 companies · FY2024–2026")

# Init chat history
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Render chat history
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "sources" in msg:
            with st.expander("Sources", expanded=False):
                for s in msg["sources"]:
                    st.caption(f"📄 {s['ticker']} · {s['filename']} · chunk {s['chunk']}")

# Handle prefill from sidebar buttons
prefill = st.session_state.pop("prefill", None)

# Chat input
user_input = st.chat_input("Ask about any SEC filing...") or prefill

if user_input:
    # Determine ticker filter
    selected_ticker = None if ticker == "All" else ticker

    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)
    st.session_state["messages"].append({"role": "user", "content": user_input})

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Retrieving from filings and generating answer..."):
            collection = get_collection()
            chunks     = retrieve(collection, user_input, ticker=selected_ticker)
            prompt     = build_prompt(user_input, chunks)
            answer     = query_ollama(prompt)

        st.markdown(answer)

        with st.expander("Sources", expanded=False):
            for s in chunks:
                st.caption(f"📄 {s['ticker']} · {s['filename']} · chunk {s['chunk']}")

    st.session_state["messages"].append({
        "role":    "assistant",
        "content": answer,
        "sources": chunks,
    })