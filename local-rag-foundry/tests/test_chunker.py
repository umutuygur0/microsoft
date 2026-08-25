import config
from src.chunker import chunk_text, document_to_chunks, parse_front_matter


def test_parse_front_matter_extracts_metadata():
    raw = "---\ntitle: Gas Leak Detection\ncategory: Safety\nid: SAFETY-001\n---\n\n# Body\ntext"
    meta, body = parse_front_matter(raw)
    assert meta == {"title": "Gas Leak Detection", "category": "Safety", "id": "SAFETY-001"}
    assert body.strip().startswith("# Body")


def test_parse_front_matter_missing_block_returns_original_text():
    raw = "# Just a heading\nsome text"
    meta, body = parse_front_matter(raw)
    assert meta == {}
    assert body == raw


def test_chunk_text_short_text_single_chunk():
    assert chunk_text("one two three", chunk_size=200, overlap=25) == ["one two three"]


def test_chunk_text_splits_and_overlaps():
    words = [f"w{i}" for i in range(50)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    assert len(chunks) > 1
    # last words of chunk 1 should reappear at the start of chunk 2 (the overlap)
    assert chunks[0].split()[-1] in chunks[1].split()


def test_document_to_chunks_uses_front_matter_and_derives_doc_id():
    raw = "---\ntitle: My Doc\ncategory: Safety\nid: DOC-1\n---\n\nSome content here."
    chunks = document_to_chunks(raw, "my-doc.md")
    assert len(chunks) == 1
    assert chunks[0].doc_id == "DOC-1"
    assert chunks[0].title == "My Doc"
    assert chunks[0].category == "Safety"


def test_document_to_chunks_falls_back_to_filename_and_h1():
    raw = "# Derived Title\n\nSome content without front matter."
    chunks = document_to_chunks(raw, "some-file.md")
    assert chunks[0].doc_id == "some-file"
    assert chunks[0].title == "Derived Title"
    assert chunks[0].category == "General"


def test_headings_stay_on_their_own_line_separate_from_following_content():
    # Regression test: chunking used to flatten all whitespace (including
    # newlines) to single spaces, so "## Source Standard\nOperations
    # Manual..." became one run-on line with no separator -- later
    # implicated in the model merging the heading's last word into the
    # content that followed it (a fabricated "Standard Operations Manual"
    # citation). A heading must now end up on its own line.
    raw = (
        "# Fire Response\n\n## Source Standard\n\n"
        "Operations Manual OM-01, Section 9 - Fire Response."
    )
    chunks = document_to_chunks(raw, "fire-response.md")
    assert len(chunks) == 1
    assert "## Source Standard\n\nOperations Manual" in chunks[0].content


def test_multiline_list_item_is_not_split_mid_sentence():
    # Regression test: an earlier version of the heading/paragraph-aware
    # chunker split a soft-wrapped bullet item at its first physical line,
    # orphaning the continuation as its own separate block.
    raw = (
        "## Key Safety Precautions\n\n"
        "- Never extinguish a gas fire unless the gas supply feeding it can also be\n"
        "isolated; doing so can create a far more dangerous situation.\n\n"
        "- Raise the alarm before attempting any first-response action.\n"
    )
    chunks = document_to_chunks(raw, "doc.md")
    assert len(chunks) == 1
    assert (
        "- Never extinguish a gas fire unless the gas supply feeding it can "
        "also be isolated; doing so can create a far more dangerous situation."
    ) in chunks[0].content


def test_degenerate_tiny_fragment_is_dropped_but_not_the_only_chunk():
    # Regression test: raw scraped reference docs sometimes retain a leftover
    # one-or-two-word line from site navigation (e.g. "Overview") as its own
    # atomic block. Live testing showed such a fragment can outscore a real
    # match in semantic search, so it must not survive as its own chunk when
    # there is other, substantial content in the same document.
    words = " ".join(f"w{i}" for i in range(config.CHUNK_SIZE + 20))
    raw = f"{words}\n\nOverview"
    chunks = document_to_chunks(raw, "doc.md")
    assert all(len(c.content.split()) >= 8 for c in chunks)
    assert not any(c.content == "Overview" for c in chunks)

    # But a document that is ONLY a tiny fragment must still produce it --
    # dropping every chunk of a document would be worse than keeping a short one.
    lone = document_to_chunks("Overview", "tiny.md")
    assert len(lone) == 1
    assert lone[0].content == "Overview"
