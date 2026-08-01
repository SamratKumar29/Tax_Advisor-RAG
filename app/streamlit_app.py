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

# ── LLM Configuration (sidebar) ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ LLM Configuration")
    st.caption("Enter your OpenRouter/NVIDIA NIM credentials. These are not stored.")

    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = ""
    if "llm_base_url" not in st.session_state:
        st.session_state.llm_base_url = "https://openrouter.ai/api/v1"
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = ""

    st.session_state.llm_api_key = st.text_input(
        "API Key",
        value=st.session_state.llm_api_key,
        type="password",
        help="Your OpenRouter or NVIDIA NIM API key",
    )
    st.session_state.llm_base_url = st.text_input(
        "Base URL",
        value=st.session_state.llm_base_url,
        help="API base URL (default: https://openrouter.ai/api/v1)",
    )
    st.session_state.llm_model = st.text_input(
        "Model",
        value=st.session_state.llm_model,
        placeholder="e.g., meta-llama/llama-3.1-8b-instruct:free",
        help="Model identifier",
    )

    st.divider()
    if st.session_state.llm_api_key and st.session_state.llm_model:
        st.success("✅ LLM configured")
    else:
        st.warning("⚠️ Please configure LLM to use the chat")

    st.divider()
    st.caption(
        "💡 Get an API key from [OpenRouter](https://openrouter.ai/keys) "
        "or [NVIDIA NIM](https://build.nvidia.com/explore/discover)."
    )
    st.caption(
        "🔒 Your credentials are only stored in this browser session "
        "and never sent to our servers."
    )

# ── Example Prompts ──────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("### 💡 Try asking:")
    example_prompts = [
        "What is the Section 80C deduction limit for FY 2026-27?",
        "Explain the new tax regime slabs for AY 2027-28",
        "What are the TDS rates for salary income?",
        "How is capital gains tax calculated on equity shares?",
        "What is the standard deduction for salaried employees?",
        "Explain Section 80D medical insurance deduction limits",
    ]
    cols = st.columns(2)
    for i, prompt in enumerate(example_prompts):
        with cols[i % 2]:
            if st.button(prompt, key=f"example_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": prompt})
                st.rerun()
    st.divider()

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
    # Check if LLM is configured
    if not st.session_state.llm_api_key or not st.session_state.llm_model:
        st.error("⚠️ Please configure your LLM credentials in the sidebar first.")
        st.stop()

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating answer..."):
            result = get_answer(
                prompt,
                api_key=st.session_state.llm_api_key,
                base_url=st.session_state.llm_base_url,
                model=st.session_state.llm_model,
            )

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