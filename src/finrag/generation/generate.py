"""Grounded answer generation. The LLM sees numbered sources and must cite them; the output is then verified."""
from __future__ import annotations

from finrag.core.schema import RetrievedChunk

from .extractive import ABSTAIN, extractive_answer
from .llm import LLM

SYSTEM = """You answer questions about SEC filings (10-K, 10-Q, 8-K) using ONLY the numbered sources provided.

Rules:
1. Use only facts stated in the sources. Never use outside knowledge or guess.
2. After every sentence that states a fact, add the source number in square brackets, e.g. [2]. Use [1][3] for several.
3. Copy numbers exactly as written. State units (the table caption gives them, e.g. "in millions") and the period (fiscal year / quarter / date).
4. Do not compute new numbers unless the user asked you to; if you must, show the inputs with their citations.
5. Table sources are Markdown: read the value from the row AND the column header that match the question. Footnotes belong to their table.
6. If the sources do not contain the answer, reply exactly: "I could not find this in the retrieved filings." and nothing else.
7. Be concise: one to three sentences unless the user asks for more. No preamble."""


def format_sources(retrieved: list[RetrievedChunk], max_chars: int = 3500) -> str:
    blocks = []
    for i, rc in enumerate(retrieved, start=1):
        c = rc.chunk
        text = c.text if len(c.text) <= max_chars else c.text[:max_chars] + "\n[truncated]"
        blocks.append(f"[{i}] {c.label} ({c.company}; period ended {c.period_end})\n{text}")
    return "\n\n".join(blocks)


def generate_answer(question: str, retrieved: list[RetrievedChunk], llm: LLM | None, standalone: str | None = None,
                    assumptions: list[str] | None = None) -> tuple[str, str]:
    """Returns (answer_text, generator_name)."""
    if not retrieved:
        return ABSTAIN, "none"
    if llm is None:
        text, _ = extractive_answer(standalone or question, retrieved)
        return text, "extractive"
    note = ("Interpretation notes:\n- " + "\n- ".join(assumptions) + "\n\n") if assumptions else ""
    user = f"{note}Sources:\n{format_sources(retrieved)}\n\nQuestion: {standalone or question}"
    return llm.complete(SYSTEM, user, max_tokens=700).strip(), llm.name
