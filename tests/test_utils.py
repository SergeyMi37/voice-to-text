"""Tests for utils module."""

import pytest
from pathlib import Path

from voice_to_text.utils import (
    get_file_hash,
    format_duration,
    estimate_cost,
)


class TestGetFileHash:
    """Tests for file hashing."""

    def test_hash_md5(self, sample_wav):
        """MD5 hash calculation."""
        hash1 = get_file_hash(sample_wav, algorithm="md5")
        hash2 = get_file_hash(sample_wav, algorithm="md5")

        assert hash1 == hash2
        assert len(hash1) == 32  # MD5 hex length

    def test_hash_sha256(self, sample_wav):
        """SHA256 hash calculation."""
        hash_value = get_file_hash(sample_wav, algorithm="sha256")

        assert len(hash_value) == 64  # SHA256 hex length

    def test_different_files_different_hash(self, sample_wav, sample_wav_stereo):
        """Different files have different hashes."""
        hash1 = get_file_hash(sample_wav)
        hash2 = get_file_hash(sample_wav_stereo)

        assert hash1 != hash2


class TestFormatDuration:
    """Tests for duration formatting."""

    def test_seconds_only(self):
        """Format seconds."""
        assert format_duration(45) == "0:45"
        assert format_duration(5) == "0:05"

    def test_minutes_and_seconds(self):
        """Format minutes and seconds."""
        assert format_duration(125) == "2:05"
        assert format_duration(3599) == "59:59"

    def test_hours(self):
        """Format with hours."""
        assert format_duration(3600) == "1:00:00"
        assert format_duration(3661) == "1:01:01"
        assert format_duration(7325) == "2:02:05"

    def test_zero(self):
        """Zero duration."""
        assert format_duration(0) == "0:00"


class TestEstimateCost:
    """Tests for API cost estimation."""

    def test_one_minute(self):
        """One minute costs $0.006."""
        cost = estimate_cost(60)
        assert cost == pytest.approx(0.006)

    def test_ten_minutes(self):
        """Ten minutes cost $0.06."""
        cost = estimate_cost(600)
        assert cost == pytest.approx(0.06)

    def test_fractional_minutes(self):
        """Fractional minutes."""
        cost = estimate_cost(90)  # 1.5 minutes
        assert cost == pytest.approx(0.009)

    def test_zero_duration(self):
        """Zero duration is free."""
        cost = estimate_cost(0)
        assert cost == 0.0