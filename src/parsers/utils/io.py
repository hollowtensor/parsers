"""File I/O and preprocessing utilities."""

from __future__ import annotations

from pathlib import Path


def resolve_path(file_path: str | Path) -> Path:
    """Resolve and validate a file path."""
    path = Path(file_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def get_file_extension(file_path: str | Path) -> str:
    """Get normalized file extension."""
    return Path(file_path).suffix.lower()


def is_image(file_path: str | Path) -> bool:
    """Check if a file is an image."""
    return get_file_extension(file_path) in {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


def is_pdf(file_path: str | Path) -> bool:
    """Check if a file is a PDF."""
    return get_file_extension(file_path) == ".pdf"
