"""Pytest fixtures and configuration."""

import pytest
from pathlib import Path
import tempfile
import wave
import struct
import os


@pytest.fixture
def temp_dir():
    """Create temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_wav(temp_dir) -> Path:
    """Create a sample WAV file with sine wave."""
    filepath = temp_dir / "test.wav"

    # Parameters
    sample_rate = 16000
    duration = 2.0  # seconds
    frequency = 440  # Hz (A4 note)

    # Generate samples
    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        # Sine wave
        value = int(32767 * 0.5 * __import__('math').sin(2 * 3.14159 * frequency * t))
        samples.append(value)

    # Write WAV
    with wave.open(str(filepath), 'w') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)  # 16-bit
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f'{len(samples)}h', *samples))

    return filepath


@pytest.fixture
def sample_wav_stereo(temp_dir) -> Path:
    """Create a stereo WAV file."""
    filepath = temp_dir / "test_stereo.wav"

    sample_rate = 44100
    duration = 1.0

    n_samples = int(sample_rate * duration)
    samples = []
    for i in range(n_samples):
        t = i / sample_rate
        left = int(32767 * 0.3 * __import__('math').sin(2 * 3.14159 * 440 * t))
        right = int(32767 * 0.3 * __import__('math').sin(2 * 3.14159 * 554 * t))
        samples.extend([left, right])

    with wave.open(str(filepath), 'w') as wav:
        wav.setnchannels(2)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(struct.pack(f'{len(samples)}h', *samples))

    return filepath


@pytest.fixture
def sample_ogg(temp_dir) -> Path:
    """Create a fake OGG file with correct magic bytes."""
    filepath = temp_dir / "test.ogg"

    # OGG magic bytes + OpusHead marker
    ogg_header = b'OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00'
    ogg_header += b'\x00\x00\x00\x00\x00\x00\x00\x00'
    ogg_header += b'\x01\x13OpusHead'

    filepath.write_bytes(ogg_header + b'\x00' * 100)

    return filepath


@pytest.fixture
def sample_mp3(temp_dir) -> Path:
    """Create a fake MP3 file with ID3 header."""
    filepath = temp_dir / "test.mp3"

    # ID3v2 header
    mp3_header = b'ID3\x04\x00\x00\x00\x00\x00\x00'
    mp3_header += b'\xff\xfb\x90\x00'  # MP3 frame header

    filepath.write_bytes(mp3_header + b'\x00' * 100)

    return filepath


@pytest.fixture
def sample_m4a(temp_dir) -> Path:
    """Create a fake M4A file with ftyp box."""
    filepath = temp_dir / "test.m4a"

    # ftyp box
    m4a_header = b'\x00\x00\x00\x18ftypM4A \x00\x00\x00\x00'
    m4a_header += b'M4A mp42isom'

    filepath.write_bytes(m4a_header + b'\x00' * 100)

    return filepath


@pytest.fixture
def voice_files_dir(temp_dir, sample_wav, sample_ogg, sample_mp3) -> Path:
    """Create directory with multiple voice files."""
    # Files already in temp_dir
    return temp_dir


@pytest.fixture
def mock_whisper_model(mocker):
    """Mock whisper model for fast tests."""
    mock = mocker.patch('whisper.load_model')
    mock_model = mocker.MagicMock()
    mock_model.transcribe.return_value = {
        'text': ' Hello, this is a test transcription.',
        'language': 'en',
        'segments': [
            {'start': 0.0, 'end': 2.0, 'text': ' Hello, this is a test transcription.'}
        ]
    }
    mock.return_value = mock_model
    return mock_model


@pytest.fixture
def mock_faster_whisper_model(mocker):
    """Mock faster-whisper model."""
    mock = mocker.patch('faster_whisper.WhisperModel')

    # Mock segment
    mock_segment = mocker.MagicMock()
    mock_segment.start = 0.0
    mock_segment.end = 2.0
    mock_segment.text = ' Hello, this is a test.'

    # Mock info
    mock_info = mocker.MagicMock()
    mock_info.language = 'en'
    mock_info.duration = 2.0
    mock_info.language_probability = 0.95

    mock_model = mocker.MagicMock()
    mock_model.transcribe.return_value = ([mock_segment], mock_info)
    mock.return_value = mock_model

    return mock_model