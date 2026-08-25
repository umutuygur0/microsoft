"""Markdown document parsing and chunking."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import config

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_FILE_EXT_RE = re.compile(r"\.(md|markdown|txt|pdf|docx)$", re.IGNORECASE)
_HEADING_RE = re.compile(r"^#{1,6}\s+.+$")
_LIST_ITEM_RE = re.compile(r"^(?:[-*+]|\d+\.)\s+.+$")


@dataclass
class Chunk:
    doc_id: str
    title: str
    category: str
    chunk_index: int
    content: str


def parse_front_matter(raw: str) -> Tuple[Dict[str, str], str]:
    """Parse a leading `--- key: value ... ---` block, if present."""
    match = _FRONT_MATTER_RE.match(raw)
    if not match:
        return {}, raw
    meta: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key:
            meta[key] = value
    return meta, raw[match.end():]


def _split_into_blocks(text: str) -> List[str]:
    """Split text into atomic blocks: every markdown heading is its own
    block; every bullet/numbered list item is its own block, including any
    soft-wrapped continuation lines up to the next blank line, heading, or
    list item; everything else is grouped into blank-line-separated
    paragraph blocks. Each block's internal whitespace is normalized to
    single spaces.

    List items are split out individually (not just headings) because a
    bulleted/numbered list has no blank line between its items, so without
    this a whole multi-step procedure or a whole bulleted precautions list
    would otherwise collapse into one large block — and later, when the
    sliding-window overlap needs to step back past it, would drag that
    entire block into the next chunk too (a near-duplicate chunk) rather
    than a modest overlap. A line-based pass (rather than regex spanning
    multiple lines) is used specifically so a list item's own soft-wrapped
    continuation lines stay merged into it instead of splitting mid-sentence
    at the item's first physical line.
    """
    blocks: List[str] = []
    current: List[str] = []

    def flush() -> None:
        if current:
            blocks.append(" ".join(" ".join(current).split()))
            current.clear()

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            flush()
            continue
        if _HEADING_RE.match(stripped):
            flush()
            blocks.append(stripped)
            continue
        if _LIST_ITEM_RE.match(stripped):
            flush()  # a new list item always starts its own block
        current.append(stripped)
    flush()
    return blocks


def _word_slices(words: List[str], chunk_size: int, overlap: int) -> List[List[str]]:
    """Raw sliding-window word slicing, used as a fallback for a single
    block that is longer than ``chunk_size`` on its own (e.g. a wall of
    scraped web text with no paragraph breaks to split on)."""
    if len(words) <= chunk_size:
        return [words]
    step = max(1, chunk_size - overlap)
    pieces: List[List[str]] = []
    start = 0
    while start < len(words):
        piece = words[start:start + chunk_size]
        if not piece:
            break
        pieces.append(piece)
        if start + chunk_size >= len(words):
            break
        start += step
    return pieces


def _atomic_units(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Blocks (headings/paragraphs), each guaranteed to be <= chunk_size words."""
    units: List[str] = []
    for block in _split_into_blocks(text):
        words = block.split()
        if len(words) <= chunk_size:
            units.append(block)
        else:
            units.extend(" ".join(piece) for piece in _word_slices(words, chunk_size, overlap))
    return units


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping chunks without breaking a heading or a
    paragraph across a chunk boundary where avoidable.

    Chunking used to work on raw whitespace-flattened words (plain
    ``text.split()``), which discarded all paragraph/heading structure — a
    heading like "## Source Standard" immediately followed by its content on
    the next line ended up as one run-on line with no separator in the
    stored chunk, e.g. "## Source Standard Operations Manual OM-01...". That
    was later implicated in the model reading a heading word as part of the
    following sentence (producing a fabricated "Standard Operations Manual"
    citation). Chunks are now built by grouping whole headings/paragraphs
    ("blocks") up to ``chunk_size`` words and joining the kept blocks with a
    blank line, so structure survives into the stored chunk content. A
    single block longer than ``chunk_size`` on its own still falls back to
    the previous raw word-window slicing so no chunk unboundedly grows.
    """
    units = _atomic_units(text, chunk_size, overlap)
    if not units:
        return []

    unit_counts = [len(u.split()) for u in units]
    if sum(unit_counts) <= chunk_size:
        return ["\n\n".join(units)]

    n = len(units)
    chunks: List[str] = []
    start = 0
    while start < n:
        count = 0
        end = start
        while end < n and (end == start or count + unit_counts[end] <= chunk_size):
            count += unit_counts[end]
            end += 1
        chunks.append("\n\n".join(units[start:end]))
        if end >= n:
            break
        # Step back from `end`, accumulating unit word counts, until at
        # least `overlap` words would repeat into the next chunk — mirrors
        # the previous raw word-window overlap, at block granularity.
        back = end
        back_count = 0
        while back > start and back_count < overlap:
            back -= 1
            back_count += unit_counts[back]
        start = back if back > start else start + 1
    return chunks


def _derive_title(body: str) -> str | None:
    match = _H1_RE.search(body)
    return match.group(1).strip() if match else None


# Below this many words, a chunk is almost certainly a degenerate fragment
# rather than real content -- observed live in the raw scraped reference
# docs, where leftover site-navigation text (e.g. a lone "Overview" or
# "Workers' Rights" left over from a page's sidebar menu) survived cleaning
# as its own tiny atomic block. Such a fragment isn't just useless noise: a
# short, generic phrase can score *higher* than a real, substantial match in
# semantic search (less content to dilute the embedding), so it can actively
# crowd out the correct chunk rather than merely sitting unused.
_MIN_CHUNK_WORDS = 8


def document_to_chunks(raw: str, file_name: str) -> List[Chunk]:
    """Turn a raw document's text into a list of structured, indexed chunks."""
    meta, body = parse_front_matter(raw)
    base_name = _FILE_EXT_RE.sub("", file_name)
    doc_id = meta.get("id") or base_name
    title = meta.get("title") or _derive_title(body) or base_name
    category = meta.get("category") or "General"

    pieces = [piece.strip() for piece in chunk_text(body) if piece.strip()]
    if len(pieces) > 1:
        substantial = [p for p in pieces if len(p.split()) >= _MIN_CHUNK_WORDS]
        pieces = substantial or pieces  # never drop every chunk of a document

    return [
        Chunk(doc_id=doc_id, title=title, category=category, chunk_index=i, content=piece)
        for i, piece in enumerate(pieces)
    ]
