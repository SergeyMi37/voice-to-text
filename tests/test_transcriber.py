"""Tests for transcriber module."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from voice_to_text.transcriber import (
    TranscriptionResult,
    Backend,
    ModelSize,
    WhisperTranscriber,
    FasterWhisperTranscriber,
    OpenAIAPITranscriber,
    create_transcriber,
)


class TestTranscriptionResult:
    """Tests for TranscriptionResult dataclass."""
    
    def test_basic_result(self):
        """Create basic transcription result."""
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            duration=1.5
        )
        
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.duration == 1.5
        assert result.segments is None
    
    def test_result_with_segments(self):
        """Result with segments for timestamps."""
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.0, "end": 2.0, "text": "world"},
        ]
        
        result = TranscriptionResult(
            text="Hello world",
            language="en",
            duration=2.0,
            segments=segments
        )
        
        assert len(result.segments) == 2
    
    def test_to_srt_empty(self):
        """SRT export without segments returns empty."""
        result = TranscriptionResult(
            text="Hello",
            language="en",
            duration=1.0,
            segments=None
        )
        
        assert result.to_srt() == ""
    
    def test_to_srt_with_segments(self):
        """SRT export with proper formatting."""
        segments = [
            {"start": 0.0, "end": 1.5, "text": " Hello"},
            {"start": 1.5, "end": 3.0, "text": " World"},
        ]
        
        result = TranscriptionResult(
            text="Hello World",
            language="en",
            duration=3.0,
            segments=segments
        )
        
        srt = result.to_srt()
        
        assert "1\n" in srt
        assert "00:00:00,000 --> 00:00:01,500" in srt
        assert "Hello" in srt
        assert "2\n" in srt
        assert "World" in srt
    
    def test_timestamp_formatting(self):
        """Timestamp formatting edge cases."""
        result = TranscriptionResult(
            text="Test",
            language="en",
            duration=3661.5,
            segments=[{"start": 3661.0, "end": 3661.5, "text": " Test"}]
        )
        
        srt = result.to_srt()
        assert "01:01:01,000" in srt


class TestBackendEnum:
    """Tests for Backend enum."""
    
    def test_backend_values(self):
        """All backends have correct values."""
        assert Backend.WHISPER.value == "whisper"
        assert Backend.FASTER_WHISPER.value == "faster"
        assert Backend.OPENAI_API.value == "api"


class TestModelSize:
    """Tests for ModelSize enum."""
    
    def test_model_sizes(self):
        """All model sizes have correct values."""
        assert ModelSize.TINY.value == "tiny"
        assert ModelSize.BASE.value == "base"
        assert ModelSize.SMALL.value == "small"
        assert ModelSize.MEDIUM.value == "medium"
        assert ModelSize.LARGE.value == "large-v3"


class TestWhisperTranscriber:
    """Tests for WhisperTranscriber."""
    
    def test_transcribe_single(self, sample_wav, mock_whisper_model):
        """Transcribe single file."""
        transcriber = WhisperTranscriber(ModelSize.BASE, device="cpu")
        
        result = transcriber.transcribe(sample_wav)
        
        assert result.text == "Hello, this is a test transcription."
        assert result.language == "en"
        assert mock_whisper_model.transcribe.called
    
    def test_transcribe_with_language(self, sample_wav, mock_whisper_model):
        """Transcribe with specified language."""
        transcriber = WhisperTranscriber(ModelSize.BASE, device="cpu")
        
        transcriber.transcribe(sample_wav, language="ru")
        
        call_args = mock_whisper_model.transcribe.call_args
        assert call_args[1].get("language") == "ru"
    
    def test_transcribe_batch(self, voice_files_dir, mock_whisper_model):
        """Batch transcription."""
        from voice_to_text.formats import find_voice_files
        
        transcriber = WhisperTranscriber(ModelSize.BASE, device="cpu")
        files = find_voice_files(voice_files_dir)
        
        results = list(transcriber.transcribe_batch(files))
        
        assert len(results) == len(files)
        for path, result in results:
            assert isinstance(result, TranscriptionResult)
    
    def test_transcribe_batch_handles_errors(self, temp_dir, mock_whisper_model):
        """Batch transcription handles individual file errors."""
        bad_file = temp_dir / "bad.wav"
        bad_file.write_bytes(b"not a wav file")
        
        mock_whisper_model.transcribe.side_effect = Exception("Decode error")
        
        transcriber = WhisperTranscriber(ModelSize.BASE, device="cpu")
        
        results = list(transcriber.transcribe_batch([bad_file]))
        
        assert len(results) == 1
        path, result = results[0]
        assert "[ERROR:" in result.text


class TestFasterWhisperTranscriber:
    """Tests for FasterWhisperTranscriber."""
    
    def test_transcribe_single(self, sample_wav, mock_faster_whisper_model):
        """Transcribe with faster-whisper."""
        transcriber = FasterWhisperTranscriber(ModelSize.BASE, device="cpu")
        
        result = transcriber.transcribe(sample_wav)
        
        assert "Hello" in result.text
        assert result.language == "en"
        assert result.confidence == 0.95
    
    def test_segments_extraction(self, sample_wav, mock_faster_whisper_model):
        """Segments are properly extracted."""
        transcriber = FasterWhisperTranscriber(ModelSize.BASE, device="cpu")
        
        result = transcriber.transcribe(sample_wav)
        
        assert result.segments is not None
        assert len(result.segments) == 1
        assert result.segments[0]["start"] == 0.0
        assert result.segments[0]["end"] == 2.0


class TestOpenAIAPITranscriber:
    """Tests for OpenAI API transcriber."""
    
    def test_transcribe_requires_api_key(self):
        """API transcriber needs API key."""
        import os
        old_key = os.environ.pop("OPENAI_API_KEY", None)
        
        try:
            with pytest.raises(Exception):
                OpenAIAPITranscriber()
        finally:
            if old_key:
                os.environ["OPENAI_API_KEY"] = old_key
    
    @patch('voice_to_text.transcriber.OpenAI')
    def test_transcribe_single(self, mock_openai, sample_wav):
        """Transcribe via API."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "API transcription result"
        mock_response.language = "en"
        mock_response.duration = 2.0
        mock_response.segments = []
        mock_client.audio.transcriptions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        transcriber = OpenAIAPITranscriber(api_key="test-key")
        result = transcriber.transcribe(sample_wav)
        
        assert result.text == "API transcription result"
        assert result.language == "en"


class TestCreateTranscriber:
    """Tests for transcriber factory."""
    
    def test_create_whisper(self, mock_whisper_model):
        """Create Whisper transcriber."""
        transcriber = create_transcriber(
            backend=Backend.WHISPER,
            model_size=ModelSize.BASE
        )
        
        assert isinstance(transcriber, WhisperTranscriber)
    
    def test_create_faster_whisper(self, mock_faster_whisper_model):
        """Create faster-whisper transcriber."""
        transcriber = create_transcriber(
            backend=Backend.FASTER_WHISPER,
            model_size=ModelSize.SMALL
        )
        
        assert isinstance(transcriber, FasterWhisperTranscriber)
    
    @patch('voice_to_text.transcriber.OpenAI')
    def test_create_api(self, mock_openai):
        """Create API transcriber."""
        transcriber = create_transcriber(
            backend=Backend.OPENAI_API,
            api_key="test-key"
        )
        
        assert isinstance(transcriber, OpenAIAPITranscriber)
    
    def test_create_invalid_backend(self):
        """Invalid backend raises error."""
        with pytest.raises(ValueError):
            create_transcriber(backend="invalid")
