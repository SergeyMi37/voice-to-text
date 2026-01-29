<<<<<<< HEAD
"""Utility functions for voice-to-text."""
=======
"""Utility functions."""
>>>>>>> 243afb8 (init commit  in github)

from pathlib import Path
import hashlib


def get_file_hash(path: Path, algorithm: str = "md5") -> str:
<<<<<<< HEAD
    """Calculate file hash for caching.
    
    Args:
        path: Path to file
        algorithm: Hash algorithm ("md5", "sha256", etc.)
        
    Returns:
        Hex digest of file hash
    """
=======
    """Calculate file hash for caching."""
>>>>>>> 243afb8 (init commit  in github)
    h = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def format_duration(seconds: float) -> str:
<<<<<<< HEAD
    """Format duration as human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string like "1:23" or "1:02:03"
    """
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    
=======
    """Format duration as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)

>>>>>>> 243afb8 (init commit  in github)
    if h > 0:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


def estimate_cost(duration_seconds: float, model: str = "whisper-1") -> float:
    """Estimate OpenAI API cost.
<<<<<<< HEAD
    
    Pricing as of 2024: $0.006 per minute
    
    Args:
        duration_seconds: Audio duration in seconds
        model: API model name
        
    Returns:
        Estimated cost in USD
    """
    minutes = duration_seconds / 60
    return minutes * 0.006


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to max length.
    
    Args:
        text: Text to truncate
        max_length: Maximum length
        suffix: Suffix to add if truncated
        
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def sanitize_filename(name: str) -> str:
    """Sanitize string for use as filename.
    
    Args:
        name: Original name
        
    Returns:
        Safe filename
    """
    # Replace problematic characters
    replacements = {
        "/": "_",
        "\\": "_",
        ":": "_",
        "*": "_",
        "?": "_",
        '"': "_",
        "<": "_",
        ">": "_",
        "|": "_",
    }
    
    result = name
    for old, new in replacements.items():
        result = result.replace(old, new)
    
    return result.strip()


def bytes_to_human(size: int) -> str:
    """Convert bytes to human-readable size.
    
    Args:
        size: Size in bytes
        
    Returns:
        Human-readable string like "1.5 MB"
    """
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"
=======

    Pricing as of 2024: $0.006 per minute
    """
    minutes = duration_seconds / 60
    return minutes * 0.006
>>>>>>> 243afb8 (init commit  in github)
