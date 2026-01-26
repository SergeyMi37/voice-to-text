"""Voice-to-Text: Bulk voice message transcription using Whisper."""

__version__ = "0.2.0"
__author__ = "Roman Gordienko"

from voice_to_text.transcriber import (
    Backend,
    ModelSize,
    TranscriptionResult,
    create_transcriber,
    BaseTranscriber,
)
from voice_to_text.formats import (
    AudioFormat,
    AudioInfo,
    detect_format,
    get_audio_info,
    find_voice_files,
    convert_to_wav,
    is_supported,
    get_supported_extensions,
)

__all__ = [
    "Backend",
    "ModelSize",
    "TranscriptionResult",
    "create_transcriber",
    "BaseTranscriber",
    "AudioFormat",
    "AudioInfo",
    "detect_format",
    "get_audio_info",
    "find_voice_files",
    "convert_to_wav",
    "is_supported",
    "get_supported_extensions",
]


# Lazy import for integrations (avoid import errors if aiogram/ptb not installed)
def __getattr__(name):
    if name == "TelegramVoiceHandler":
        from voice_to_text.integrations.telegram import TelegramVoiceHandler
        return TelegramVoiceHandler
    if name == "transcribe_telegram_voice":
        from voice_to_text.integrations.telegram import transcribe_telegram_voice
        return transcribe_telegram_voice
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
