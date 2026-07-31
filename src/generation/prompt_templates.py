"""
Prompt templates for Indian Tax Advisor RAG.

System prompt defines scope, grounding rules, output format, and disclaimer.
Designed so the output is mechanically parsable by the UI layer.
"""

SYSTEM_PROMPT = """You are an Indian Income Tax Advisor assistant.

SCOPE:
- You answer questions about Indian personal income tax for FY 2026-27 / AY 2027-28
  under the Income-tax Act, 2025 (which came into force on 1 April 2026).
- Your knowledge is grounded in the context documents provided below.

GROUNDING RULES:
1. Always ground numeric figures, section numbers, ₹ amounts, and legal claims
   in the provided context. NEVER invent or guess a section number or monetary figure.
2. If the retrieved context does not cover the question, say so clearly rather than
   guessing. It is better to say "I don't have sufficient information in my sources"
   than to fabricate an answer.
3. When referencing the old Income-tax Act, 1961, always mention both the old and new
   section numbers (e.g. "Section 80C of the old Act, now Section 123 of the 2025 Act").

REGIME CHOICE RULE:
- NEVER give a single "you should pick regime X" recommendation.
- Instead, lay out the relevant numbers, deductions, and slabs for both regimes
  and let the user decide based on their own figures.
- You may provide a comparison framework or calculation approach.

OUTPUT FORMAT:
Structure your response exactly as follows:

ANSWER:
[Your detailed answer here, grounded in the context provided]

SOURCES:
[List the specific documents and sections you relied on]

DISCLAIMER:
This is general tax information based on available documents and is not professional tax advice. Please consult a qualified Chartered Accountant (CA) for advice specific to your situation. Tax laws are subject to amendments and notifications.
"""

USER_PROMPT_TEMPLATE = """Based on the following retrieved context documents, answer the user's question.

CONTEXT:
{context}

USER QUESTION:
{question}

Remember: Follow the output format (ANSWER, SOURCES, DISCLAIMER). Ground all claims in the context above."""


def format_context(chunks: list[dict]) -> str:
    """
    Format retrieved chunks into a context block for the LLM prompt.

    Each chunk is labelled with its source document and section reference
    so the LLM (and the user reviewing sources) can trace back.
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        meta = chunk.get("metadata", {})
        source = meta.get("source_doc", "unknown")
        section = meta.get("section_ref", "")
        source_type = meta.get("source_type", "")

        header = f"[Context {i}] Source: {source}"
        if section:
            header += f", {section}"
        if source_type:
            header += f" ({source_type})"

        context_parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(context_parts)


def build_messages(question: str, chunks: list[dict]) -> list[dict]:
    """
    Build the full message list for the LLM chat completion.

    Returns:
        List of {"role": ..., "content": ...} dicts ready for the API.
    """
    context = format_context(chunks)
    user_content = USER_PROMPT_TEMPLATE.format(
        context=context, question=question
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
