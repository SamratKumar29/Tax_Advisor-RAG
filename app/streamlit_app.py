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

from src.config import LLM_MODEL, MAX_QUERY_CHARS, IndexesMissingError, check_indexes
from src.generation.chain import answer as get_answer
from src.llm_client import LLMConfigError, LLMUnavailableError

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
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None

# ── LLM Configuration (sidebar) ─────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ LLM Configuration")
    st.caption("Enter your OpenRouter/NVIDIA NIM credentials. These are not stored.")

    if "llm_api_key" not in st.session_state:
        st.session_state.llm_api_key = ""
    if "llm_base_url" not in st.session_state:
        st.session_state.llm_base_url = "https://openrouter.ai/api/v1"
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = LLM_MODEL

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
                st.session_state.pending_prompt = prompt
                st.rerun()
    st.divider()

# ── Rendering helpers ────────────────────────────────────────────────────
def render_citation_check(check: dict | None) -> None:
    """Surface deterministic citation problems (see src.generation.citations)."""
    if not check:
        return
    if check.get("invalid_markers"):
        st.warning(
            "⚠️ The answer cites context blocks that were not provided: "
            + ", ".join(f"[{n}]" for n in check["invalid_markers"])
        )
    for item in check.get("unsupported", []):
        st.warning(
            f"⚠️ {item['claim']} is not found in the cited source(s) "
            + ", ".join(f"[{n}]" for n in item["cited"]) + ". Verify before relying on it."
        )
    if check.get("uncited_figures"):
        st.caption(f"ℹ️ {len(check['uncited_figures'])} sentence(s) state a figure without a citation.")


def render_assistant_details(msg: dict) -> None:
    """Model-reported sources, citation check, retrieved chunks and disclaimer."""
    if msg.get("sources_text"):
        st.markdown("**Sources (as reported by model):**")
        st.markdown(msg["sources_text"])
    render_citation_check(msg.get("citation_check"))

    with st.expander("📚 Retrieved Sources (ground truth)", expanded=False):
        for i, chunk in enumerate(msg.get("retrieved_chunks", []), 1):
            label = f"**[{i}] {chunk['source_doc']}**"
            if chunk.get("section_ref"):
                label += f" · {chunk['section_ref']}"
            if chunk.get("page"):
                label += f" · p.{chunk['page']}"
            label += f" · _{chunk['source_type']}_ · topic: `{chunk['topic']}`"
            st.markdown(label)
            st.text(chunk["text"])
            st.divider()

    st.caption(msg.get("disclaimer", ""))
    if msg.get("model_used"):
        st.caption(f"Model: `{msg['model_used']}`")


# ── Render chat history ──────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and "retrieved_chunks" in msg:
            render_assistant_details(msg)

# ── Chat input ───────────────────────────────────────────────────────────
def handle_prompt(prompt: str) -> None:
    """Single answer path shared by the example buttons and the chat input."""
    if not st.session_state.llm_api_key or not st.session_state.llm_model:
        st.error("⚠️ Please configure your LLM credentials in the sidebar first.")
        return
    if len(prompt) > MAX_QUERY_CHARS:
        st.error(f"⚠️ Question is too long (max {MAX_QUERY_CHARS} characters).")
        return

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate answer
    with st.chat_message("assistant"):
        try:
            check_indexes()
            with st.spinner("Searching documents and generating answer..."):
                result = get_answer(
                    prompt,
                    api_key=st.session_state.llm_api_key,
                    base_url=st.session_state.llm_base_url,
                    model=st.session_state.llm_model,
                )
        except (LLMUnavailableError, IndexesMissingError, ValueError) as e:
            # Remove the orphaned user turn so history stays consistent.
            st.session_state.messages.pop()
            if isinstance(e, LLMConfigError):
                st.error(f"⚠️ {e}")
            elif isinstance(e, LLMUnavailableError):
                st.error(
                    "⚠️ The language model is unavailable right now (tried: "
                    f"{', '.join(e.models_tried) or 'none'}). Please retry or pick another model."
                )
            else:
                st.error(f"⚠️ {e}")
            return

        message = {
            "role": "assistant",
            "content": result["answer"],
            "sources_text": result.get("sources_text", ""),
            "citation_check": result.get("citation_check"),
            "retrieved_chunks": result["retrieved_chunks"],
            "disclaimer": result["disclaimer"],
            "model_used": result.get("model_used", ""),
        }
        st.markdown(message["content"])
        render_assistant_details(message)

    st.session_state.messages.append(message)


typed_prompt = st.chat_input("Ask a tax question (e.g. 'What is the Section 80C limit?')")
pending_prompt = st.session_state.pending_prompt
st.session_state.pending_prompt = None
if prompt := (typed_prompt or pending_prompt):
    handle_prompt(prompt)
