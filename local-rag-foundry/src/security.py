"""Upload safety checks: filename sanitisation, path-traversal, and size guards."""
from __future__ import annotations

import re
from pathlib import PurePosixPath

import config

_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class UploadRejected(Exception):
    """Raised when an uploaded file fails a safety check."""


def validate_upload(filename: str, size_bytes: int) -> str:
    """Validate an uploaded file's name and size.

    Returns the sanitised, safe filename to use for ingestion. Raises
    ``UploadRejected`` with a user-facing reason if the file must not be ingested.
    """
    if not filename:
        raise UploadRejected("Missing file name.")

    # PurePosixPath(...).name strips any directory components. If the result
    # differs from the input, the original contained a path separator
    # (e.g. "../../etc/passwd" or "sub/dir/file.md") — reject outright rather
    # than silently truncating it.
    name = PurePosixPath(filename).name
    if name != filename or not _SAFE_NAME_RE.match(name):
        raise UploadRejected("Filename must be a plain name with no path separators or special characters.")

    suffix = "." + name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if suffix not in config.ALLOWED_UPLOAD_EXTENSIONS:
        allowed = ", ".join(config.ALLOWED_UPLOAD_EXTENSIONS)
        raise UploadRejected(f"Unsupported file type '{suffix or '(none)'}'. Allowed: {allowed}")

    if size_bytes > config.MAX_UPLOAD_SIZE_BYTES:
        limit_mb = config.MAX_UPLOAD_SIZE_BYTES / (1024 * 1024)
        actual_mb = size_bytes / (1024 * 1024)
        raise UploadRejected(f"File too large ({actual_mb:.2f} MB). Limit is {limit_mb:.0f} MB.")

    return name
