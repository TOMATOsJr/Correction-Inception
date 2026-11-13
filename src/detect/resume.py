"""Resumable CSV outputs: a job killed before its first flush leaves a 0-byte file, which pandas cannot read."""
from pathlib import Path


def has_rows(path: Path) -> bool:
    """True when the file exists and holds at least a header (a 0-byte file counts as missing)."""
    return path.exists() and path.stat().st_size > 0
