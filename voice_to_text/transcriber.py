"""Core transcription engine with multiple backends."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterator
import tempfile
import os


class Backend(Enum):
    """Available transcription backends."""
    WHISPER = "whisper"           # openai-whisper (local)
    FASTER_WHISPER = "faster"     # faster-whisper (CTranslate2)
    OPENAI_API = "api"            # OpenAI API (cloud)
    GROQ = "groq"                 # Groq API (fast, free tier)


class ModelSize(Enum):
    """Whisper model sizes."""
    TINY = "tiny"         # 39M params, ~1GB VRAM
    BASE = "base"         # 74M params, ~1GB VRAM
    SMALL = "small"       # 244M params, ~2GB VRAM
    MEDIUM = "medium"     # 769M params, ~5GB VRAM
    LARGE = "large-v3"    # 1550M params, ~10GB VRAM


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""
    text: str
    language: str
    duration: float
    segments: list[dict] | None = None
    confidence: float | None = None
    
    def to_srt(self) -> str:
        """Export as SRT subtitles."""
        if not self.segments:
            return ""
        
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start = self._format_timestamp(seg["start"])
            end = self._format_timestamp(seg["end"])
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(seg["text"].strip())
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class BaseTranscriber(ABC):
    """Abstract base class for transcribers."""
    
    @abstractmethod
    def transcribe(
        self, 
        audio_path: Path, 
        language: str | None = None
    ) -> TranscriptionResult:
        """Transcribe a single audio file."""
        pass
    
    @abstractmethod
    def transcribe_batch(
        self, 
        audio_paths: list[Path],
        language: str | None = None
    ) -> Iterator[tuple[Path, TranscriptionResult]]:
        """Transcribe multiple audio files."""
        pass


class WhisperTranscriber(BaseTranscriber):
    """Local Whisper transcription using openai-whisper."""
    
    def __init__(
        self, 
        model_size: ModelSize = ModelSize.BASE,
        device: str = "auto"
    ):
        import whisper
        import torch
        
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = whisper.load_model(model_size.value, device=device)
        
    def transcribe(
        self, 
        audio_path: Path, 
        language: str | None = None
    ) -> TranscriptionResult:
        """Transcribe a single audio file."""
        audio_path = self._ensure_compatible(audio_path)
        
        options = {}
        if language:
            options["language"] = language
        
        result = self.model.transcribe(str(audio_path), **options)
        
        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", "unknown"),
            duration=result["segments"][-1]["end"] if result["segments"] else 0,
            segments=result["segments"],
        )
    
    def transcribe_batch(
        self, 
        audio_paths: list[Path],
        language: str | None = None
    ) -> Iterator[tuple[Path, TranscriptionResult]]:
        """Transcribe multiple audio files."""
        for path in audio_paths:
            try:
                result = self.transcribe(path, language)
                yield path, result
            except Exception as e:
                yield path, TranscriptionResult(
                    text=f"[ERROR: {e}]",
                    language="",
                    duration=0
                )
    
    def _ensure_compatible(self, path: Path) -> Path:
        """Convert to WAV if format is not directly supported."""
        suffix = path.suffix.lower()
        
        # Whisper handles these natively
        if suffix in {".wav", ".mp3", ".m4a", ".flac"}:
            return path
        
        # Convert OGG/OPUS (Telegram) and others
        if suffix in {".ogg", ".opus", ".webm", ".amr"}:
            return self._convert_to_wav(path)
        
        return path
    
    def _convert_to_wav(self, path: Path) -> Path:
        """Convert audio to WAV using pydub."""
        from pydub import AudioSegment
        
        audio = AudioSegment.from_file(str(path))
        
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(tmp.name, format="wav")
        
        return Path(tmp.name)


class FasterWhisperTranscriber(BaseTranscriber):
    """Faster Whisper using CTranslate2 - 4x faster on CPU."""
    
    def __init__(
        self,
        model_size: ModelSize = ModelSize.BASE,
        device: str = "auto",
        compute_type: str = "auto"
    ):
        from faster_whisper import WhisperModel
        
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"
        
        self.model = WhisperModel(
            model_size.value,
            device=device,
            compute_type=compute_type
        )
    
    def transcribe(
        self, 
        audio_path: Path, 
        language: str | None = None
    ) -> TranscriptionResult:
        """Transcribe a single audio file."""
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True,
        )
        
        segments_list = []
        full_text = []
        
        for seg in segments:
            segments_list.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            })
            full_text.append(seg.text)
        
        return TranscriptionResult(
            text=" ".join(full_text).strip(),
            language=info.language,
            duration=info.duration,
            segments=segments_list,
            confidence=info.language_probability,
        )
    
    def transcribe_batch(
        self, 
        audio_paths: list[Path],
        language: str | None = None
    ) -> Iterator[tuple[Path, TranscriptionResult]]:
        """Transcribe multiple audio files."""
        for path in audio_paths:
            try:
                result = self.transcribe(path, language)
                yield path, result
            except Exception as e:
                yield path, TranscriptionResult(
                    text=f"[ERROR: {e}]",
                    language="",
                    duration=0
                )


class OpenAIAPITranscriber(BaseTranscriber):
    """OpenAI Whisper API - no GPU needed, pay per use."""
    
    def __init__(self, api_key: str | None = None):
        from openai import OpenAI
        
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def transcribe(
        self, 
        audio_path: Path, 
        language: str | None = None
    ) -> TranscriptionResult:
        """Transcribe a single audio file via API."""
        with open(audio_path, "rb") as f:
            kwargs = {"model": "whisper-1", "file": f}
            if language:
                kwargs["language"] = language
            
            kwargs["response_format"] = "verbose_json"
            
            response = self.client.audio.transcriptions.create(**kwargs)
        
        return TranscriptionResult(
            text=response.text,
            language=response.language,
            duration=response.duration,
            segments=[
                {"start": s.start, "end": s.end, "text": s.text}
                for s in (response.segments or [])
            ]
        )
    
    def transcribe_batch(
        self, 
        audio_paths: list[Path],
        language: str | None = None
    ) -> Iterator[tuple[Path, TranscriptionResult]]:
        """Transcribe multiple audio files via API."""
        for path in audio_paths:
            try:
                result = self.transcribe(path, language)
                yield path, result
            except Exception as e:
                yield path, TranscriptionResult(
                    text=f"[ERROR: {e}]",
                    language="",
                    duration=0
                )


class GroqTranscriber(BaseTranscriber):
    """Groq Whisper API - extremely fast, free tier available.
    
    Groq provides one of the fastest Whisper inference APIs.
    Free tier: 20 requests/min, 2000 requests/day (as of 2024).
    
    Supported models:
    - whisper-large-v3 (default, best quality)
    - whisper-large-v3-turbo (faster, slightly lower quality)
    """
    
    SUPPORTED_MODELS = {"whisper-large-v3", "whisper-large-v3-turbo"}
    MAX_FILE_SIZE = 25 * 1024 * 1024  # 25 MB limit
    
    def __init__(
        self, 
        api_key: str | None = None,
        model: str = "whisper-large-v3-turbo"
    ):
        from groq import Groq
        
        self.client = Groq(api_key=api_key or os.getenv("GROQ_API_KEY"))
        self.model = model
        
        if model not in self.SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model: {model}. "
                f"Supported: {', '.join(self.SUPPORTED_MODELS)}"
            )
    
    def transcribe(
        self, 
        audio_path: Path, 
        language: str | None = None
    ) -> TranscriptionResult:
        """Transcribe a single audio file via Groq API."""
        # Check file size
        file_size = audio_path.stat().st_size
        if file_size > self.MAX_FILE_SIZE:
            raise ValueError(
                f"File too large: {file_size / 1024 / 1024:.1f} MB. "
                f"Groq limit: {self.MAX_FILE_SIZE / 1024 / 1024:.0f} MB"
            )
        
        with open(audio_path, "rb") as f:
            kwargs = {
                "model": self.model,
                "file": f,
                "response_format": "verbose_json",
            }
            if language:
                kwargs["language"] = language
            
            response = self.client.audio.transcriptions.create(**kwargs)
        
        # Parse segments from response
        segments = []
        if hasattr(response, 'segments') and response.segments:
            for seg in response.segments:
                segments.append({
                    "start": seg.get("start", seg.start) if hasattr(seg, 'start') else seg["start"],
                    "end": seg.get("end", seg.end) if hasattr(seg, 'end') else seg["end"],
                    "text": seg.get("text", seg.text) if hasattr(seg, 'text') else seg["text"],
                })
        
        return TranscriptionResult(
            text=response.text,
            language=getattr(response, 'language', language or 'unknown'),
            duration=getattr(response, 'duration', 0) or 0,
            segments=segments if segments else None,
        )
    
    def transcribe_batch(
        self, 
        audio_paths: list[Path],
        language: str | None = None
    ) -> Iterator[tuple[Path, TranscriptionResult]]:
        """Transcribe multiple audio files via Groq API."""
        for path in audio_paths:
            try:
                result = self.transcribe(path, language)
                yield path, result
            except Exception as e:
                yield path, TranscriptionResult(
                    text=f"[ERROR: {e}]",
                    language="",
                    duration=0
                )


def create_transcriber(
    backend: Backend = Backend.FASTER_WHISPER,
    model_size: ModelSize = ModelSize.BASE,
    **kwargs
) -> BaseTranscriber:
    """Factory function to create a transcriber.
    
    Args:
        backend: Transcription backend to use
        model_size: Model size (for local backends)
        **kwargs: Additional arguments passed to transcriber
        
    Returns:
        Configured transcriber instance
    """
    if backend == Backend.WHISPER:
        return WhisperTranscriber(model_size, **kwargs)
    elif backend == Backend.FASTER_WHISPER:
        return FasterWhisperTranscriber(model_size, **kwargs)
    elif backend == Backend.OPENAI_API:
        return OpenAIAPITranscriber(**kwargs)
    elif backend == Backend.GROQ:
        return GroqTranscriber(**kwargs)
    else:
        raise ValueError(f"Unknown backend: {backend}")
