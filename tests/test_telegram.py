"""Tests for Telegram integration."""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from pathlib import Path
import tempfile
import asyncio

from voice_to_text.integrations.telegram import (
    TelegramVoiceHandler,
    TelegramTranscriptionResult,
    TelegramMessageType,
    transcribe_telegram_voice,
)
from voice_to_text.transcriber import (
    TranscriptionResult,
    Backend,
    ModelSize,
)


class TestTelegramTranscriptionResult:
    """Tests for TelegramTranscriptionResult."""
    
    def test_basic_properties(self):
        """Test basic property access."""
        inner = TranscriptionResult(
            text="Hello world",
            language="en",
            duration=5.5,
        )
        
        result = TelegramTranscriptionResult(
            result=inner,
            message_type=TelegramMessageType.VOICE,
            file_id="abc123",
            file_size=1024,
            duration=5,
        )
        
        assert result.text == "Hello world"
        assert result.language == "en"
        assert result.file_id == "abc123"
        assert result.file_size == 1024
        assert result.duration == 5
        assert result.message_type == TelegramMessageType.VOICE
    
    def test_format_reply_simple(self):
        """Test format_reply without stats."""
        inner = TranscriptionResult(text="Test text", language="ru", duration=10.0)
        result = TelegramTranscriptionResult(
            result=inner,
            message_type=TelegramMessageType.VOICE,
            file_id="test",
        )
        
        reply = result.format_reply(include_stats=False)
        assert reply == "Test text"
    
    def test_format_reply_with_stats(self):
        """Test format_reply with stats."""
        inner = TranscriptionResult(text="Test text", language="ru", duration=65.0)
        result = TelegramTranscriptionResult(
            result=inner,
            message_type=TelegramMessageType.VOICE,
            file_id="test",
        )
        
        reply = result.format_reply(include_stats=True)
        assert "Test text" in reply
        assert "🌐 RU" in reply
        assert "⏱ 1:05" in reply


class TestTelegramVoiceHandler:
    """Tests for TelegramVoiceHandler."""
    
    def test_init_default(self):
        """Test default initialization."""
        handler = TelegramVoiceHandler()
        
        assert handler.backend == Backend.FASTER_WHISPER
        assert handler.model_size == ModelSize.BASE
        assert handler.default_language is None
        assert handler.cleanup is True
    
    def test_init_custom(self):
        """Test custom initialization."""
        handler = TelegramVoiceHandler(
            backend=Backend.OPENAI_API,
            model_size=ModelSize.SMALL,
            language="ru",
            temp_dir="/tmp/test",
            cleanup=False,
        )
        
        assert handler.backend == Backend.OPENAI_API
        assert handler.model_size == ModelSize.SMALL
        assert handler.default_language == "ru"
        assert handler.temp_dir == Path("/tmp/test")
        assert handler.cleanup is False
    
    def test_get_extension_from_filename(self):
        """Test _get_extension with filename."""
        ext = TelegramVoiceHandler._get_extension("test.ogg", None, ".mp3")
        assert ext == ".ogg"
        
        ext = TelegramVoiceHandler._get_extension("audio.wav", "audio/mpeg", ".mp3")
        assert ext == ".wav"
    
    def test_get_extension_from_mime(self):
        """Test _get_extension with MIME type."""
        ext = TelegramVoiceHandler._get_extension(None, "audio/ogg", ".mp3")
        assert ext == ".ogg"
        
        ext = TelegramVoiceHandler._get_extension(None, "audio/mpeg", ".wav")
        assert ext == ".mp3"
    
    def test_get_extension_default(self):
        """Test _get_extension fallback to default."""
        ext = TelegramVoiceHandler._get_extension(None, None, ".mp3")
        assert ext == ".mp3"
        
        ext = TelegramVoiceHandler._get_extension("", "unknown/type", ".wav")
        assert ext == ".wav"


class TestTelegramVoiceHandlerAsync:
    """Async tests for TelegramVoiceHandler."""
    
    @pytest.fixture
    def handler(self):
        """Create handler with mocked transcriber."""
        handler = TelegramVoiceHandler()
        
        # Mock transcriber
        mock_transcriber = MagicMock()
        mock_transcriber.transcribe.return_value = TranscriptionResult(
            text="Transcribed text",
            language="ru",
            duration=5.0,
        )
        handler._transcriber = mock_transcriber
        
        return handler
    
    @pytest.fixture
    def mock_bot_aiogram(self):
        """Create mock aiogram bot."""
        bot = AsyncMock()
        
        # Mock file info
        mock_file = MagicMock()
        mock_file.file_path = "voice/file_123.ogg"
        bot.get_file.return_value = mock_file
        
        # Mock download
        async def mock_download(file_path, destination):
            # Create fake file
            Path(destination).write_bytes(b"fake audio data")
        
        bot.download_file = mock_download
        
        return bot
    
    @pytest.fixture
    def mock_voice_message(self):
        """Create mock voice message."""
        message = MagicMock()
        message.voice = MagicMock()
        message.voice.file_id = "voice_123"
        message.voice.file_size = 1024
        message.voice.duration = 5
        message.video_note = None
        message.audio = None
        return message
    
    @pytest.mark.asyncio
    async def test_transcribe_voice_aiogram(self, handler, mock_bot_aiogram, mock_voice_message):
        """Test voice transcription with aiogram."""
        result = await handler.transcribe_voice(
            mock_voice_message,
            mock_bot_aiogram,
        )
        
        assert result.text == "Transcribed text"
        assert result.language == "ru"
        assert result.message_type == TelegramMessageType.VOICE
        assert result.file_id == "voice_123"
    
    @pytest.mark.asyncio
    async def test_transcribe_message_auto_detect(self, handler, mock_bot_aiogram, mock_voice_message):
        """Test auto message type detection."""
        result = await handler.transcribe_message(
            mock_voice_message,
            mock_bot_aiogram,
        )
        
        assert result is not None
        assert result.message_type == TelegramMessageType.VOICE
    
    @pytest.mark.asyncio
    async def test_transcribe_message_no_audio(self, handler, mock_bot_aiogram):
        """Test with message without audio."""
        message = MagicMock()
        message.voice = None
        message.video_note = None
        message.audio = None
        
        result = await handler.transcribe_message(message, mock_bot_aiogram)
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_transcribe_voice_no_attachment(self, handler, mock_bot_aiogram):
        """Test error when no voice attachment."""
        message = MagicMock()
        message.voice = None
        
        with pytest.raises(ValueError, match="no voice attachment"):
            await handler.transcribe_voice(message, mock_bot_aiogram)


class TestConvenienceFunction:
    """Tests for convenience functions."""
    
    @pytest.mark.asyncio
    async def test_transcribe_telegram_voice_function(self):
        """Test transcribe_telegram_voice convenience function."""
        # This would require full integration test with mocked dependencies
        # For now, just verify the function exists and has correct signature
        import inspect
        
        sig = inspect.signature(transcribe_telegram_voice)
        params = list(sig.parameters.keys())
        
        assert "file_id" in params
        assert "bot" in params
        assert "language" in params
        assert "backend" in params
        assert "model_size" in params
