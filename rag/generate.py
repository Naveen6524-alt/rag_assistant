"""Answer generation with an LLM, grounded in retrieved chunks.

Requires:  pip install anthropic   and   ANTHROPIC_API_KEY in your environment.
Optional:  ANTHROPIC_MODEL (default below).

NOTE: this module was written but NOT run against the live API in the environment
where the project was scaffolded (no API key there). Run `python cli.py ask --generate ...`
yourself and read the outputs before you describe this part as working.
"""
import os

DEFAULT_MODEL = "claude-sonnet-5-5"

SYSTEM = (
    "You answer questions about software documentation using ONLY the numbered context "
    "passages provided. Cite the passages you used like [1] or [2][3]. If the passages do "
    "not contain the answer, say you could not find it in the documentation instead of "
    "guessing. Keep answers short and concrete."
)


def build_context(chunks) -> str:
    parts = []
    for n, c in enumerate(chunks, start=1):
        parts.append(f"[{n}] ({c.source} > {c.heading})\n{c.text}")
    return "\n\n".join(parts)


def answer(question: str, chunks, model: str | None = None, max_tokens: int = 700) -> str:
    try:
        import anthropic
    except ImportError as e:
        raise ImportError("pip install anthropic") from e
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable first.")
    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model or os.environ.get("ANTHROPIC_MODEL", DEFAULT_MODEL),
        max_tokens=max_tokens,
        system=SYSTEM,
        messages=[{"role": "user",
                   "content": f"Context passages:\n\n{build_context(chunks)}\n\nQuestion: {question}"}],
    )
    return "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
