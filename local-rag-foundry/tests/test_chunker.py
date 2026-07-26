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
