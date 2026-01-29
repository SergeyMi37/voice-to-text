"""
Telegram Bot Integration for Voice-to-Text
==========================================

Provides seamless voice message transcription for Telegram bots.

Usage with aiogram 3.x:
    from voice_to_text.integrations.telegram import TelegramVoiceHandler
    
    handler = TelegramVoiceHandler()
    
    @dp.message(F.voice)
    async def handle_voice(message: Message, bot: Bot):
        result = await handler.transcribe_voice(message, bot)
        await message.reply(result.text)

Usage with python-telegram-bot:
    from voice_to_text.integrations.telegram import TelegramVoiceHandler
    
    handler = TelegramVoiceHandler()
    
    async def voice_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
        result = await handler.transcribe_from_ptb(update.message, context.bot)
        await update.message.reply_text(result.text)
"""

from __future__ import annotations

import asyncio
import tempfile
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable
import os

from ..transcriber import (
    create_transcriber,
    Backend,
    ModelSize,
    TranscriptionResult,
    BaseTranscriber,
)

if TYPE_CHECKING:
    # Avoid import errors if aiogram/ptb not installed
    pass


class TelegramMessageType(Enum):
    """Types of Telegram messages that can be transcribed."""
    VOICE = "voice"           # Voice message (OGG Opus)
    VIDEO_NOTE = "video_note"  # Round video message
    AUDIO = "audio"           # Audio file
    DOCUMENT = "document"     # Document (if audio)


@dataclass
class TelegramTranscriptionResult:
    """Extended result with Telegram-specific metadata."""
    result: TranscriptionResult
    message_type: TelegramMessageType
    file_id: str
    file_size: int | None = None
    duration: int | None = None  # From Telegram metadata
    
    @property
    def text(self) -> str:
        return self.result.text
    
    @property
    def language(self) -> str:
        return self.result.language
    
    def format_reply(self, include_stats: bool = False) -> str:
        """Format result for Telegram reply."""
        text = self.result.text
        
        if include_stats:
            stats = []
            if self.result.language:
                stats.append(f"🌐 {self.result.language.upper()}")
            if self.result.duration:
                mins = int(self.result.duration // 60)
                secs = int(self.result.duration % 60)
                stats.append(f"⏱ {mins}:{secs:02d}")
            
            if stats:
                text += f"\n\n{' · '.join(stats)}"
        
        return text


class TelegramVoiceHandler:
    """
    Handler for transcribing Telegram voice messages.
    
    Supports:
    - Voice messages (OGG Opus) - standard voice
    - Video notes (round videos) - extracts audio
    - Audio files
    - Documents (if audio format)
    
    Args:
        backend: Transcription backend (default: faster-whisper)
        model_size: Whisper model size
        language: Default language (None = auto-detect)
        temp_dir: Directory for temporary files
        cleanup: Auto-delete temp files after transcription
        
    Example:
        handler = TelegramVoiceHandler(
            backend=Backend.FASTER_WHISPER,
            model_size=ModelSize.SMALL,
            language="ru"
        )
    """
    
    def __init__(
        self,
        backend: Backend = Backend.FASTER_WHISPER,
        model_size: ModelSize = ModelSize.BASE,
        language: str | None = None,
        temp_dir: Path | str | None = None,
        cleanup: bool = True,
        transcriber: BaseTranscriber | None = None,
    ):
        self.backend = backend
        self.model_size = model_size
        self.default_language = language
        self.temp_dir = Path(temp_dir) if temp_dir else None
        self.cleanup = cleanup
        
        # Lazy load transcriber
        self._transcriber = transcriber
        self._transcriber_lock = asyncio.Lock()
    
    async def get_transcriber(self) -> BaseTranscriber:
        """Get or create transcriber (lazy loading)."""
        if self._transcriber is None:
            async with self._transcriber_lock:
                if self._transcriber is None:
                    # Run in thread pool (model loading is blocking)
                    loop = asyncio.get_event_loop()
                    self._transcriber = await loop.run_in_executor(
                        None,
                        lambda: create_transcriber(
                            backend=self.backend,
                            model_size=self.model_size,
                        )
                    )
        return self._transcriber
    
    # ==================== AIOGRAM 3.x ====================
    
    async def transcribe_voice(
        self,
        message: Any,  # aiogram.types.Message
        bot: Any,      # aiogram.Bot
        language: str | None = None,
    ) -> TelegramTranscriptionResult:
        """
        Transcribe voice message (aiogram 3.x).
        
        Args:
            message: Telegram message with voice
            bot: aiogram Bot instance
            language: Override default language
            
        Returns:
            TelegramTranscriptionResult
        """
        if not message.voice:
            raise ValueError("Message has no voice attachment")
        
        voice = message.voice
        file_path = await self._download_file_aiogram(voice.file_id, bot, ".ogg")
        
        try:
            result = await self._transcribe_file(
                file_path,
                language or self.default_language
            )
            
            return TelegramTranscriptionResult(
                result=result,
                message_type=TelegramMessageType.VOICE,
                file_id=voice.file_id,
                file_size=voice.file_size,
                duration=voice.duration,
            )
        finally:
            if self.cleanup and file_path.exists():
                file_path.unlink()
    
    async def transcribe_video_note(
        self,
        message: Any,  # aiogram.types.Message
        bot: Any,      # aiogram.Bot
        language: str | None = None,
    ) -> TelegramTranscriptionResult:
        """
        Transcribe video note (round video) - aiogram 3.x.
        
        Extracts audio from video note and transcribes.
        """
        if not message.video_note:
            raise ValueError("Message has no video_note attachment")
        
        video_note = message.video_note
        file_path = await self._download_file_aiogram(
            video_note.file_id, bot, ".mp4"
        )
        
        try:
            # Extract audio (video notes are MP4)
            audio_path = await self._extract_audio(file_path)
            
            try:
                result = await self._transcribe_file(
                    audio_path,
                    language or self.default_language
                )
                
                return TelegramTranscriptionResult(
                    result=result,
                    message_type=TelegramMessageType.VIDEO_NOTE,
                    file_id=video_note.file_id,
                    file_size=video_note.file_size,
                    duration=video_note.duration,
                )
            finally:
                if self.cleanup and audio_path.exists():
                    audio_path.unlink()
        finally:
            if self.cleanup and file_path.exists():
                file_path.unlink()
    
    async def transcribe_audio(
        self,
        message: Any,  # aiogram.types.Message
        bot: Any,      # aiogram.Bot
        language: str | None = None,
    ) -> TelegramTranscriptionResult:
        """Transcribe audio file attachment (aiogram 3.x)."""
        if not message.audio:
            raise ValueError("Message has no audio attachment")
        
        audio = message.audio
        ext = self._get_extension(audio.file_name, audio.mime_type, ".mp3")
        file_path = await self._download_file_aiogram(audio.file_id, bot, ext)
        
        try:
            result = await self._transcribe_file(
                file_path,
                language or self.default_language
            )
            
            return TelegramTranscriptionResult(
                result=result,
                message_type=TelegramMessageType.AUDIO,
                file_id=audio.file_id,
                file_size=audio.file_size,
                duration=audio.duration,
            )
        finally:
            if self.cleanup and file_path.exists():
                file_path.unlink()
    
    async def transcribe_message(
        self,
        message: Any,  # aiogram.types.Message
        bot: Any,      # aiogram.Bot
        language: str | None = None,
    ) -> TelegramTranscriptionResult | None:
        """
        Auto-detect message type and transcribe (aiogram 3.x).
        
        Handles: voice, video_note, audio
        Returns None if message has no transcribable content.
        """
        if message.voice:
            return await self.transcribe_voice(message, bot, language)
        elif message.video_note:
            return await self.transcribe_video_note(message, bot, language)
        elif message.audio:
            return await self.transcribe_audio(message, bot, language)
        
        return None
    
    async def _download_file_aiogram(
        self,
        file_id: str,
        bot: Any,
        extension: str,
    ) -> Path:
        """Download file from Telegram (aiogram)."""
        file = await bot.get_file(file_id)
        
        # Create temp file
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            fd, path = tempfile.mkstemp(suffix=extension, dir=self.temp_dir)
            os.close(fd)
            file_path = Path(path)
        else:
            fd, path = tempfile.mkstemp(suffix=extension)
            os.close(fd)
            file_path = Path(path)
        
        # Download
        await bot.download_file(file.file_path, file_path)
        
        return file_path
    
    # ==================== PYTHON-TELEGRAM-BOT ====================
    
    async def transcribe_from_ptb(
        self,
        message: Any,  # telegram.Message
        bot: Any,      # telegram.Bot
        language: str | None = None,
    ) -> TelegramTranscriptionResult | None:
        """
        Transcribe voice/audio from python-telegram-bot.
        
        Args:
            message: telegram.Message
            bot: telegram.Bot
            language: Override language
            
        Returns:
            TelegramTranscriptionResult or None
        """
        if message.voice:
            return await self._transcribe_ptb_voice(message, bot, language)
        elif message.video_note:
            return await self._transcribe_ptb_video_note(message, bot, language)
        elif message.audio:
            return await self._transcribe_ptb_audio(message, bot, language)
        
        return None
    
    async def _transcribe_ptb_voice(
        self,
        message: Any,
        bot: Any,
        language: str | None,
    ) -> TelegramTranscriptionResult:
        """Transcribe voice (python-telegram-bot)."""
        voice = message.voice
        file = await bot.get_file(voice.file_id)
        
        file_path = await self._download_file_ptb(file, ".ogg")
        
        try:
            result = await self._transcribe_file(
                file_path,
                language or self.default_language
            )
            
            return TelegramTranscriptionResult(
                result=result,
                message_type=TelegramMessageType.VOICE,
                file_id=voice.file_id,
                file_size=voice.file_size,
                duration=voice.duration,
            )
        finally:
            if self.cleanup and file_path.exists():
                file_path.unlink()
    
    async def _transcribe_ptb_video_note(
        self,
        message: Any,
        bot: Any,
        language: str | None,
    ) -> TelegramTranscriptionResult:
        """Transcribe video note (python-telegram-bot)."""
        video_note = message.video_note
        file = await bot.get_file(video_note.file_id)
        
        file_path = await self._download_file_ptb(file, ".mp4")
        
        try:
            audio_path = await self._extract_audio(file_path)
            
            try:
                result = await self._transcribe_file(
                    audio_path,
                    language or self.default_language
                )
                
                return TelegramTranscriptionResult(
                    result=result,
                    message_type=TelegramMessageType.VIDEO_NOTE,
                    file_id=video_note.file_id,
                    file_size=video_note.file_size,
                    duration=video_note.duration,
                )
            finally:
                if self.cleanup and audio_path.exists():
                    audio_path.unlink()
        finally:
            if self.cleanup and file_path.exists():
                file_path.unlink()
    
    async def _transcribe_ptb_audio(
        self,
        message: Any,
        bot: Any,
        language: str | None,
    ) -> TelegramTranscriptionResult:
        """Transcribe audio (python-telegram-bot)."""
        audio = message.audio
        file = await bot.get_file(audio.file_id)
        
        ext = self._get_extension(audio.file_name, audio.mime_type, ".mp3")
        file_path = await self._download_file_ptb(file, ext)
        
        try:
            result = await self._transcribe_file(
                file_path,
                language or self.default_language
            )
            
            return TelegramTranscriptionResult(
                result=result,
                message_type=TelegramMessageType.AUDIO,
                file_id=audio.file_id,
                file_size=audio.file_size,
                duration=audio.duration,
            )
        finally:
            if self.cleanup and file_path.exists():
                file_path.unlink()
    
    async def _download_file_ptb(
        self,
        file: Any,  # telegram.File
        extension: str,
    ) -> Path:
        """Download file (python-telegram-bot)."""
        if self.temp_dir:
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            fd, path = tempfile.mkstemp(suffix=extension, dir=self.temp_dir)
            os.close(fd)
            file_path = Path(path)
        else:
            fd, path = tempfile.mkstemp(suffix=extension)
            os.close(fd)
            file_path = Path(path)
        
        await file.download_to_drive(file_path)
        
        return file_path
    
    # ==================== COMMON METHODS ====================
    
    async def transcribe_file_id(
        self,
        file_id: str,
        bot: Any,
        extension: str = ".ogg",
        language: str | None = None,
        framework: str = "aiogram",
    ) -> TranscriptionResult:
        """
        Transcribe by file_id directly.
        
        Args:
            file_id: Telegram file_id
            bot: Bot instance
            extension: File extension hint
            language: Language override
            framework: "aiogram" or "ptb"
            
        Returns:
            TranscriptionResult
        """
        if framework == "aiogram":
            file_path = await self._download_file_aiogram(file_id, bot, extension)
        else:
            file = await bot.get_file(file_id)
            file_path = await self._download_file_ptb(file, extension)
        
        try:
            return await self._transcribe_file(
                file_path,
                language or self.default_language
            )
        finally:
            if self.cleanup and file_path.exists():
                file_path.unlink()
    
    async def _transcribe_file(
        self,
        file_path: Path,
        language: str | None,
    ) -> TranscriptionResult:
        """Run transcription in thread pool."""
        transcriber = await self.get_transcriber()
        
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: transcriber.transcribe(file_path, language=language)
        )
        
        return result
    
    async def _extract_audio(self, video_path: Path) -> Path:
        """Extract audio from video file."""
        audio_path = video_path.with_suffix(".wav")
        
        # Try pydub first
        try:
            from pydub import AudioSegment
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: AudioSegment.from_file(str(video_path)).export(
                    str(audio_path), format="wav"
                )
            )
            return audio_path
        except ImportError:
            pass
        
        # Fallback to ffmpeg
        proc = await asyncio.create_subprocess_exec(
            "ffmpeg", "-i", str(video_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            str(audio_path),
            "-y",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await proc.wait()
        
        if not audio_path.exists():
            raise RuntimeError(f"Failed to extract audio from {video_path}")
        
        return audio_path
    
    @staticmethod
    def _get_extension(
        filename: str | None,
        mime_type: str | None,
        default: str,
    ) -> str:
        """Determine file extension."""
        if filename:
            ext = Path(filename).suffix.lower()
            if ext:
                return ext
        
        if mime_type:
            mime_map = {
                "audio/ogg": ".ogg",
                "audio/opus": ".opus",
                "audio/mpeg": ".mp3",
                "audio/mp3": ".mp3",
                "audio/wav": ".wav",
                "audio/x-wav": ".wav",
                "audio/mp4": ".m4a",
                "audio/m4a": ".m4a",
                "audio/aac": ".aac",
                "audio/flac": ".flac",
                "video/mp4": ".mp4",
            }
            ext = mime_map.get(mime_type.lower())
            if ext:
                return ext
        
        return default


# ==================== DECORATOR FACTORIES ====================

def voice_handler(
    handler_instance: TelegramVoiceHandler | None = None,
    language: str | None = None,
    reply_with_text: bool = True,
    include_stats: bool = False,
    error_message: str = "❌ Не удалось распознать голосовое сообщение",
):
    """
    Decorator factory for aiogram voice handlers.
    
    Usage:
        handler = TelegramVoiceHandler()
        
        @dp.message(F.voice)
        @voice_handler(handler)
        async def handle_voice(message: Message, bot: Bot, result: TelegramTranscriptionResult):
            # result is already transcribed
            print(result.text)
    """
    def decorator(func: Callable):
        async def wrapper(message, bot, **kwargs):
            h = handler_instance or TelegramVoiceHandler()
            
            try:
                result = await h.transcribe_voice(message, bot, language)
                
                if reply_with_text:
                    await message.reply(result.format_reply(include_stats))
                
                return await func(message, bot, result=result, **kwargs)
                
            except Exception as e:
                if error_message:
                    await message.reply(error_message)
                raise
        
        return wrapper
    return decorator


# ==================== CONVENIENCE FUNCTIONS ====================

async def transcribe_telegram_voice(
    file_id: str,
    bot: Any,
    language: str | None = None,
    backend: Backend = Backend.FASTER_WHISPER,
    model_size: ModelSize = ModelSize.BASE,
) -> TranscriptionResult:
    """
    Quick function to transcribe a voice message by file_id.
    
    Example:
        result = await transcribe_telegram_voice(
            message.voice.file_id,
            bot,
            language="ru"
        )
        print(result.text)
    """
    handler = TelegramVoiceHandler(
        backend=backend,
        model_size=model_size,
        language=language,
    )
    
    return await handler.transcribe_file_id(file_id, bot, ".ogg", language)
