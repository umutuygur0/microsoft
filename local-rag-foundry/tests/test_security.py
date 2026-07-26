import pytest

from src.security import UploadRejected, validate_upload


def test_valid_markdown_file_is_accepted():
    assert validate_upload("my-notes.md", 1024) == "my-notes.md"


def test_valid_text_file_is_accepted():
    assert validate_upload("faq.txt", 1024) == "faq.txt"


@pytest.mark.parametrize("bad_name", [
    "../../etc/passwd",
    "../secrets.md",
    "sub/dir/file.md",
    "sub\\dir\\file.md",
])
def test_path_traversal_attempts_are_rejected(bad_name):
    with pytest.raises(UploadRejected):
        validate_upload(bad_name, 1024)


def test_disallowed_extension_is_rejected():
    with pytest.raises(UploadRejected):
        validate_upload("script.py", 1024)


def test_no_extension_is_rejected():
    with pytest.raises(UploadRejected):
        validate_upload("noextension", 1024)


def test_oversized_file_is_rejected():
    import config
    with pytest.raises(UploadRejected):
        validate_upload("big.md", config.MAX_UPLOAD_SIZE_BYTES + 1)


def test_empty_filename_is_rejected():
    with pytest.raises(UploadRejected):
        validate_upload("", 1024)


def test_special_characters_in_filename_are_rejected():
    with pytest.raises(UploadRejected):
        validate_upload("weird;name|here.md", 1024)
