"""Markdown document parsing and chunking."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Tuple

import config

_FRONT_MATTER_RE = re.compile(r"^---\s*\n(.*?)\n---\s*\n?", re.DOTALL)
_H1_RE = re.compile(r"^#\s+(.+)$", re.MULTILINE)
_FILE_EXT_RE = re.compile(r"\.(md|markdown|txt)$", re.IGNORECASE)


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


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []
    if len(words) <= chunk_size:
        return [" ".join(words)]

    step = max(1, chunk_size - overlap)
    chunks: List[str] = []
    start = 0
    while start < len(words):
        piece = words[start:start + chunk_size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if start + chunk_size >= len(words):
            break
        start += step
    return chunks


def _derive_title(body: str) -> str | None:
    match = _H1_RE.search(body)
    return match.group(1).strip() if match else None


def document_to_chunks(raw: str, file_name: str) -> List[Chunk]:
    """Turn a raw document's text into a list of structured, indexed chunks."""
    meta, body = parse_front_matter(raw)
    base_name = _FILE_EXT_RE.sub("", file_name)
    doc_id = meta.get("id") or base_name
    title = meta.get("title") or _derive_title(body) or base_name
    category = meta.get("category") or "General"
    return [
        Chunk(doc_id=doc_id, title=title, category=category, chunk_index=i, content=piece.strip())
        for i, piece in enumerate(chunk_text(body))
        if piece.strip()
    ]
