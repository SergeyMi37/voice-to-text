"""Voice-to-Text: Bulk voice message transcription using Whisper."""

__version__ = "0.1.0"
__author__ = "Roman"

from voice_to_text.transcriber import (
    Backend,
    ModelSize,
    TranscriptionResult,
    create_transcriber,
)
from voice_to_text.formats import (
    AudioFormat,
    AudioInfo,
    detect_format,
    get_audio_info,
    find_voice_files,
)

__all__ = [
    "Backend",
    "ModelSize",
    "TranscriptionResult",
    "create_transcriber",
    "AudioFormat",
    "AudioInfo",
    "detect_format",
    "get_audio_info",
    "find_voice_files",
]
