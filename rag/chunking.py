"""Markdown-aware chunking.

Design choices (each one is something you should be able to defend):
- Split on headings first, so a chunk never straddles two topics.
- Then pack paragraphs up to `max_words`, carrying `overlap` words into the next chunk.
- Keep the heading path ("Page > Section") as metadata. It can be prepended to the
  indexed text, which is a toggle you can ablate in the evaluation.
"""
import re
from dataclasses import dataclass
from pathlib import Path

INCLUDE_RE = re.compile(r"^\{\*.*\*\}\s*$")          # {* ../../docs_src/x.py hl[1] *}
ANCHOR_RE = re.compile(r"\s*\{\s*#[\w\-]+\s*\}\s*$")  # heading anchor: { #some-id }
ADMONITION_RE = re.compile(r"^\s*///.*$")             # /// note | Title   and   ///
HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")


@dataclass
class Chunk:
    id: str
    source: str   # path relative to the corpus root
    heading: str  # "H1 > H2 > H3"
    text: str

    def index_text(self, prepend_heading: bool = True) -> str:
        return f"{self.heading}\n{self.text}" if prepend_heading and self.heading else self.text


def _clean(line: str) -> str:
    return ANCHOR_RE.sub("", line.rstrip())


def split_sections(markdown: str):
    """Yield (heading_path, body_text). Ignores '#' inside code fences."""
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    in_fence = False

    def current_path():
        return " > ".join(t for _, t in stack)

    for raw in markdown.splitlines():
        if raw.strip().startswith("```"):
            in_fence = not in_fence
        if not in_fence:
            if INCLUDE_RE.match(raw) or ADMONITION_RE.match(raw):
                continue
            m = HEADING_RE.match(raw)
            if m:
                text = "\n".join(body).strip()
                if text:
                    yield current_path(), text
                body = []
                level, title = len(m.group(1)), _clean(m.group(2)).strip()
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
                continue
        body.append(_clean(raw))
    text = "\n".join(body).strip()
    if text:
        yield current_path(), text


def pack_paragraphs(text: str, max_words: int, overlap: int) -> list[str]:
    """Greedy-pack paragraphs into chunks of <= max_words, with word overlap."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current: list[str] = []
    count = 0
    for para in paragraphs:
        words = para.split()
        # a single huge paragraph: split it by words
        while len(words) > max_words:
            if current:
                chunks.append("\n\n".join(current)); current, count = [], 0
            chunks.append(" ".join(words[:max_words]))
            words = words[max(max_words - overlap, 1):]
        para = " ".join(words)
        if not words:
            continue
        if count + len(words) > max_words and current:
            chunks.append("\n\n".join(current))
            tail = " ".join("\n\n".join(current).split()[-overlap:]) if overlap else ""
            current, count = ([tail] if tail else []), len(tail.split())
        current.append(para)
        count += len(words)
    if current:
        chunks.append("\n\n".join(current))
    return chunks


def chunk_file(path: Path, root: Path, max_words: int = 180, overlap: int = 30,
               min_words: int = 8) -> list[Chunk]:
    rel = path.relative_to(root).as_posix()
    out: list[Chunk] = []
    for heading, body in split_sections(path.read_text(encoding="utf-8")):
        for piece in pack_paragraphs(body, max_words, overlap):
            if len(piece.split()) >= min_words:
                out.append(Chunk(id=f"{rel}::{len(out)}", source=rel, heading=heading, text=piece))
    return out


def load_corpus(root: str | Path, max_words: int = 180, overlap: int = 30) -> list[Chunk]:
    root = Path(root)
    files = sorted(root.rglob("*.md"))
    if not files:
        raise FileNotFoundError(f"No .md files found under {root}")
    chunks: list[Chunk] = []
    for f in files:
        chunks.extend(chunk_file(f, root, max_words, overlap))
    return chunks
