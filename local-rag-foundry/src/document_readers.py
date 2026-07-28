"""Best-effort plain-text extraction from PDF and DOCX sources.

Used by both the ingestion pipeline (docs/ folder) and the Streamlit upload
handler (in-memory bytes). Extraction failures never raise past this module —
callers get an empty string and can show a friendly warning instead of crashing
(e.g. a scanned/image-only PDF with no text layer, which is out of scope: that
would require OCR, not just PDF parsing).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Union

Source = Union[str, Path, bytes]


def _as_stream(source: Source) -> io.BytesIO | str:
    if isinstance(source, (bytes, bytearray)):
        return io.BytesIO(source)
    return str(source)


def extract_pdf_text(source: Source) -> str:
    """Extract text from a PDF (file path or raw bytes), page by page.

    Returns an empty string if the file cannot be parsed or has no
    extractable text (e.g. a scanned image with no OCR text layer).
    """
    try:
        from pypdf import PdfReader

        reader = PdfReader(_as_stream(source))
        pages = []
        for page in reader.pages:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001 - a single bad page shouldn't lose the rest
                continue
        return "\n\n".join(p for p in pages if p).strip()
    except Exception:  # noqa: BLE001 - corrupt/unsupported PDF: treat as "no text"
        return ""


_DOCX_HEADING_PREFIXES = {
    "Title": "# ",
    "Heading 1": "# ",
    "Heading 2": "## ",
    "Heading 3": "### ",
}


def extract_docx_text(source: Source) -> str:
    """Extract text from a DOCX file (file path or raw bytes), paragraph by paragraph.

    Paragraphs styled as Title/Heading 1-3 are converted to Markdown headings
    (#, ##, ###) so ``src/chunker.py``'s existing title-detection logic (which
    looks for a leading Markdown H1) works on DOCX uploads the same way it
    already does for .md files, instead of falling back to the raw filename.

    Tables and images are not extracted, only plain paragraph text. Returns an
    empty string if the file cannot be parsed.
    """
    try:
        from docx import Document

        document = Document(_as_stream(source))
        lines = []
        for p in document.paragraphs:
            text = p.text.strip()
            if not text:
                continue
            prefix = _DOCX_HEADING_PREFIXES.get(p.style.name if p.style else "", "")
            lines.append(prefix + text)
        return "\n\n".join(lines).strip()
    except Exception:  # noqa: BLE001 - corrupt/unsupported DOCX: treat as "no text"
        return ""


def extract_text(filename: str, source: Source) -> str:
    """Dispatch to the right extractor based on the file's extension.

    Markdown and plain text are handled by the caller (they are just decoded
    as UTF-8); this dispatcher only covers the binary formats.
    """
    suffix = Path(filename).suffix.lower()
    if suffix == ".pdf":
        return extract_pdf_text(source)
    if suffix == ".docx":
        return extract_docx_text(source)
    raise ValueError(f"No text extractor registered for '{suffix}' files.")
