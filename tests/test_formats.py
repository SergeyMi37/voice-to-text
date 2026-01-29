"""Tests for formats module."""

import pytest
from pathlib import Path

from voice_to_text.formats import (
    AudioFormat,
    AudioInfo,
    detect_format,
    get_supported_extensions,
    find_voice_files,
    is_supported,
    convert_to_wav,
)


class TestAudioFormat:
    """Tests for AudioFormat enum."""
    
    def test_voice_message_formats(self):
        """Telegram and WhatsApp formats should be recognized as voice."""
        info = AudioInfo(format=AudioFormat.OGG_OPUS)
        assert info.is_voice_message is True
        
        info = AudioInfo(format=AudioFormat.M4A)
        assert info.is_voice_message is True
    
    def test_non_voice_formats(self):
        """Regular audio formats are not voice messages."""
        info = AudioInfo(format=AudioFormat.MP3)
        assert info.is_voice_message is False
        
        info = AudioInfo(format=AudioFormat.WAV)
        assert info.is_voice_message is False


class TestDetectFormat:
    """Tests for format detection."""
    
    def test_detect_wav(self, sample_wav):
        """Detect WAV format."""
        fmt = detect_format(sample_wav)
        assert fmt == AudioFormat.WAV
    
    def test_detect_ogg(self, sample_ogg):
        """Detect OGG Opus format."""
        fmt = detect_format(sample_ogg)
        assert fmt == AudioFormat.OGG_OPUS
    
    def test_detect_mp3(self, sample_mp3):
        """Detect MP3 format."""
        fmt = detect_format(sample_mp3)
        assert fmt == AudioFormat.MP3
    
    def test_detect_m4a(self, sample_m4a):
        """Detect M4A format."""
        fmt = detect_format(sample_m4a)
        assert fmt == AudioFormat.M4A
    
    def test_detect_unknown(self, temp_dir):
        """Unknown format for random data."""
        filepath = temp_dir / "random.xyz"
        filepath.write_bytes(b'\x00\x01\x02\x03\x04\x05')
        
        fmt = detect_format(filepath)
        assert fmt == AudioFormat.UNKNOWN
    
    def test_detect_by_extension_fallback(self, temp_dir):
        """Fallback to extension when magic bytes don't match."""
        filepath = temp_dir / "fake.flac"
        filepath.write_bytes(b'\x00\x00\x00\x00')
        
        fmt = detect_format(filepath)
        assert fmt == AudioFormat.FLAC


class TestSupportedExtensions:
    """Tests for supported extensions."""
    
    def test_all_expected_extensions(self):
        """All common voice formats are supported."""
        extensions = get_supported_extensions()
        
        assert ".ogg" in extensions
        assert ".opus" in extensions
        assert ".mp3" in extensions
        assert ".wav" in extensions
        assert ".m4a" in extensions
        assert ".webm" in extensions
        assert ".amr" in extensions
    
    def test_is_supported(self, sample_wav, sample_ogg):
        """Check is_supported function."""
        assert is_supported(sample_wav) is True
        assert is_supported(sample_ogg) is True
    
    def test_is_not_supported(self, temp_dir):
        """Unsupported files return False."""
        filepath = temp_dir / "document.pdf"
        filepath.write_bytes(b'%PDF-1.4')
        
        assert is_supported(filepath) is False


class TestFindVoiceFiles:
    """Tests for finding voice files."""
    
    def test_find_in_directory(self, voice_files_dir):
        """Find all voice files in directory."""
        files = find_voice_files(voice_files_dir)
        
        assert len(files) >= 3
        extensions = {f.suffix.lower() for f in files}
        assert ".wav" in extensions
        assert ".ogg" in extensions
        assert ".mp3" in extensions
    
    def test_find_recursive(self, temp_dir, sample_wav):
        """Find files recursively."""
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        
        new_path = subdir / "nested.wav"
        new_path.write_bytes(sample_wav.read_bytes())
        
        files_recursive = find_voice_files(temp_dir, recursive=True)
        nested_found_recursive = any("nested" in f.name for f in files_recursive)
        
        assert nested_found_recursive is True
    
    def test_empty_directory(self, temp_dir):
        """Empty directory returns empty list."""
        empty_dir = temp_dir / "empty"
        empty_dir.mkdir()
        
        files = find_voice_files(empty_dir)
        assert files == []


class TestConvertToWav:
    """Tests for WAV conversion."""
    
    def test_convert_wav_to_wav(self, sample_wav, temp_dir):
        """Converting WAV to WAV should work."""
        output = temp_dir / "output.wav"
        
        result = convert_to_wav(sample_wav, output)
        
        assert result.exists()
        assert result.stat().st_size > 0
    
    def test_convert_with_resample(self, sample_wav_stereo, temp_dir):
        """Convert stereo to mono 16kHz."""
        output = temp_dir / "output_mono.wav"
        
        result = convert_to_wav(
            sample_wav_stereo, 
            output,
            sample_rate=16000,
            channels=1
        )
        
        assert result.exists()
        
        import wave
        with wave.open(str(result), 'r') as wav:
            assert wav.getnchannels() == 1
            assert wav.getframerate() == 16000
    
    def test_convert_creates_temp_file(self, sample_wav):
        """Without output path, creates temp file."""
        result = convert_to_wav(sample_wav, output_path=None)
        
        assert result.exists()
        assert result.suffix == ".wav"
        
        result.unlink()
