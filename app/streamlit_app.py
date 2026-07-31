"""
Streamlit Chat UI for Indian Tax Advisor RAG.

Keeps UI free of business logic — only calls src.generation.chain.answer()
and renders the result with retrieved sources and disclaimer.
"""

import sys
from pathlib import Path

# Add project root to path so imports work when running via `streamlit run`
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from src.generation.chain import answer as get_answer

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Indian Tax Advisor",
    page_icon="🏛️",
    layout="centered",
)

st.title("🏛️ Indian Tax Advisor")
st.caption(
    "FY 2026-27 / AY 2027-28 · Income-tax Act, 2025 · "
    "Hybrid RAG (Dense + BM25 + RRF)"
)

# ── Session state ────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Render chat history ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        # Show sources expander for assistant messages
        if msg["role"] == "assistant" and "retrieved_chunks" in msg:
            with st.expander("📚 Retrieved Sources (ground truth)", expanded=False):
                for i, chunk in enumerate(msg["retrieved_chunks"], 1):
                    source = chunk["source_doc"]
                    section = chunk.get("section_ref", "")
                    stype = chunk["source_type"]
                    topic = chunk["topic"]

                    label = f"**[{i}] {source}**"
                    if section:
                        label += f" · {section}"
                    label += f" · _{stype}_ · topic: `{topic}`"

                    st.markdown(label)
                    st.text(chunk["text"])
                    st.divider()

            st.caption(msg.get("disclaimer", ""))

# ── Chat input ───────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a tax question (e.g. 'What is the Section 80C limit?')"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = get_answer(prompt)

        # Display answer
        st.markdown(result["answer"])

        # Display LLM-reported sources if any
        if result.get("sources_text"):
            st.markdown("**Sources (as reported by model):**")
            st.markdown(result["sources_text"])

        # Display actual retrieved chunks (ground truth)
        with st.expander("📚 Retrieved Sources (ground truth)", expanded=False):
            for i, chunk in enumerate(result["retrieved_chunks"], 1):
                source = chunk["source_doc"]
                section = chunk.get("section_ref", "")
                stype = chunk["source_type"]
                topic = chunk["topic"]

                label = f"**[{i}] {source}**"
                if section:
                    label += f" · {section}"
                label += f" · _{stype}_ · topic: `{topic}`"

                st.markdown(label)
                st.text(chunk["text"])
                st.divider()

        # Disclaimer footer
        st.caption(result["disclaimer"])

    # Save to session state
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "retrieved_chunks": result["retrieved_chunks"],
        "disclaimer": result["disclaimer"],
    })
