"""Tests for utils module."""

import pytest
from pathlib import Path

from voice_to_text.utils import (
    get_file_hash,
    format_duration,
    estimate_cost,
<<<<<<< HEAD
    truncate_text,
    sanitize_filename,
    bytes_to_human,
=======
>>>>>>> 243afb8 (init commit  in github)
)


class TestGetFileHash:
    """Tests for file hashing."""
<<<<<<< HEAD
    
=======

>>>>>>> 243afb8 (init commit  in github)
    def test_hash_md5(self, sample_wav):
        """MD5 hash calculation."""
        hash1 = get_file_hash(sample_wav, algorithm="md5")
        hash2 = get_file_hash(sample_wav, algorithm="md5")
<<<<<<< HEAD
        
        assert hash1 == hash2
        assert len(hash1) == 32
    
    def test_hash_sha256(self, sample_wav):
        """SHA256 hash calculation."""
        hash_value = get_file_hash(sample_wav, algorithm="sha256")
        
        assert len(hash_value) == 64
    
=======

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex length

    def test_hash_sha256(self, sample_wav):
        """SHA256 hash calculation."""
        hash_value = get_file_hash(sample_wav, algorithm="sha256")

        assert len(hash_value) == 64  # SHA256 hex length

>>>>>>> 243afb8 (init commit  in github)
    def test_different_files_different_hash(self, sample_wav, sample_wav_stereo):
        """Different files have different hashes."""
        hash1 = get_file_hash(sample_wav)
        hash2 = get_file_hash(sample_wav_stereo)
<<<<<<< HEAD
        
=======

>>>>>>> 243afb8 (init commit  in github)
        assert hash1 != hash2


class TestFormatDuration:
    """Tests for duration formatting."""
<<<<<<< HEAD
    
=======

>>>>>>> 243afb8 (init commit  in github)
    def test_seconds_only(self):
        """Format seconds."""
        assert format_duration(45) == "0:45"
        assert format_duration(5) == "0:05"
<<<<<<< HEAD
    
=======

>>>>>>> 243afb8 (init commit  in github)
    def test_minutes_and_seconds(self):
        """Format minutes and seconds."""
        assert format_duration(125) == "2:05"
        assert format_duration(3599) == "59:59"
<<<<<<< HEAD
    
=======

>>>>>>> 243afb8 (init commit  in github)
    def test_hours(self):
        """Format with hours."""
        assert format_duration(3600) == "1:00:00"
        assert format_duration(3661) == "1:01:01"
        assert format_duration(7325) == "2:02:05"
<<<<<<< HEAD
    
=======

>>>>>>> 243afb8 (init commit  in github)
    def test_zero(self):
        """Zero duration."""
        assert format_duration(0) == "0:00"


class TestEstimateCost:
    """Tests for API cost estimation."""
<<<<<<< HEAD
    
=======

>>>>>>> 243afb8 (init commit  in github)
    def test_one_minute(self):
        """One minute costs $0.006."""
        cost = estimate_cost(60)
        assert cost == pytest.approx(0.006)
<<<<<<< HEAD
    
=======

>>>>>>> 243afb8 (init commit  in github)
    def test_ten_minutes(self):
        """Ten minutes cost $0.06."""
        cost = estimate_cost(600)
        assert cost == pytest.approx(0.06)
<<<<<<< HEAD
    
    def test_fractional_minutes(self):
        """Fractional minutes."""
        cost = estimate_cost(90)
        assert cost == pytest.approx(0.009)
    
    def test_zero_duration(self):
        """Zero duration is free."""
        cost = estimate_cost(0)
        assert cost == 0.0


class TestTruncateText:
    """Tests for text truncation."""
    
    def test_short_text_unchanged(self):
        """Short text is not truncated."""
        text = "Hello"
        result = truncate_text(text, max_length=100)
        assert result == "Hello"
    
    def test_long_text_truncated(self):
        """Long text is truncated with suffix."""
        text = "Hello world, this is a long text"
        result = truncate_text(text, max_length=15)
        assert result == "Hello world,..."
        assert len(result) == 15
    
    def test_custom_suffix(self):
        """Custom suffix works."""
        text = "Hello world"
        result = truncate_text(text, max_length=8, suffix="…")
        assert result == "Hello w…"


class TestSanitizeFilename:
    """Tests for filename sanitization."""
    
    def test_safe_name_unchanged(self):
        """Safe names are unchanged."""
        assert sanitize_filename("hello.txt") == "hello.txt"
    
    def test_removes_slashes(self):
        """Slashes are replaced."""
        assert sanitize_filename("path/to/file") == "path_to_file"
        assert sanitize_filename("path\\to\\file") == "path_to_file"
    
    def test_removes_special_chars(self):
        """Special characters are replaced."""
        assert sanitize_filename('file:name?.txt') == "file_name_.txt"
        assert sanitize_filename('file<>|name') == "file___name"
    
    def test_strips_whitespace(self):
        """Whitespace is stripped."""
        assert sanitize_filename("  hello  ") == "hello"


class TestBytesToHuman:
    """Tests for human-readable byte sizes."""
    
    def test_bytes(self):
        """Small sizes in bytes."""
        assert bytes_to_human(500) == "500.0 B"
    
    def test_kilobytes(self):
        """Kilobyte sizes."""
        assert bytes_to_human(1024) == "1.0 KB"
        assert bytes_to_human(1536) == "1.5 KB"
    
    def test_megabytes(self):
        """Megabyte sizes."""
        assert bytes_to_human(1048576) == "1.0 MB"
        assert bytes_to_human(5 * 1024 * 1024) == "5.0 MB"
    
    def test_gigabytes(self):
        """Gigabyte sizes."""
        assert bytes_to_human(1073741824) == "1.0 GB"
=======

    def test_fractional_minutes(self):
        """Fractional minutes."""
        cost = estimate_cost(90)  # 1.5 minutes
        assert cost == pytest.approx(0.009)

    def test_zero_duration(self):
        """Zero duration is free."""
        cost = estimate_cost(0)
        assert cost == 0.0
>>>>>>> 243afb8 (init commit  in github)
