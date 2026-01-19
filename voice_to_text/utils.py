"""Utility functions."""

from pathlib import Path
import hashlib


def get_file_hash(path: Path, algorithm: str = "md5") -> str:
    """Calculate file hash for caching."""
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def format_duration(seconds: float) -> str:
    """Format duration as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def estimate_cost(duration_seconds: float, model: str = "whisper-1") -> float:
    """Estimate OpenAI API cost.

    Pricing as of 2024: $0.006 per minute
    """
    minutes = duration_seconds / 60
    return minutes * 0.006