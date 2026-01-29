<<<<<<< HEAD
"""Core transcription engine with multiple backends."""

=======
>>>>>>> 243afb8 (init commit  in github)
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
<<<<<<< HEAD
from typing import Iterator
=======
from typing import Optional, Iterator
>>>>>>> 243afb8 (init commit  in github)
import tempfile
import os


class Backend(Enum):
<<<<<<< HEAD
    """Available transcription backends."""
    WHISPER = "whisper"           # openai-whisper (local)
    FASTER_WHISPER = "faster"     # faster-whisper (CTranslate2)
    OPENAI_API = "api"            # OpenAI API (cloud)


class ModelSize(Enum):
    """Whisper model sizes."""
    TINY = "tiny"         # 39M params, ~1GB VRAM
    BASE = "base"         # 74M params, ~1GB VRAM
    SMALL = "small"       # 244M params, ~2GB VRAM
    MEDIUM = "medium"     # 769M params, ~5GB VRAM
    LARGE = "large-v3"    # 1550M params, ~10GB VRAM
=======
    WHISPER = "whisper"  # openai-whisper (локальный)
    FASTER_WHISPER = "faster"  # faster-whisper (CTranslate2)
    OPENAI_API = "api"  # OpenAI API (облако)


class ModelSize(Enum):
    TINY = "tiny"  # 39M params, ~1GB VRAM
    BASE = "base"  # 74M params, ~1GB VRAM
    SMALL = "small"  # 244M params, ~2GB VRAM
    MEDIUM = "medium"  # 769M params, ~5GB VRAM
    LARGE = "large-v3"  # 1550M params, ~10GB VRAM
>>>>>>> 243afb8 (init commit  in github)


@dataclass
class TranscriptionResult:
<<<<<<< HEAD
    """Result of audio transcription."""
    text: str
    language: str
    duration: float
    segments: list[dict] | None = None
    confidence: float | None = None
    
=======
    text: str
    language: str
    duration: float
    segments: list[dict] | None = None  # timestamps если нужны
    confidence: float | None = None

>>>>>>> 243afb8 (init commit  in github)
    def to_srt(self) -> str:
        """Export as SRT subtitles."""
        if not self.segments:
            return ""
<<<<<<< HEAD
        
=======

>>>>>>> 243afb8 (init commit  in github)
        lines = []
        for i, seg in enumerate(self.segments, 1):
            start = self._format_timestamp(seg["start"])
            end = self._format_timestamp(seg["end"])
            lines.append(f"{i}")
            lines.append(f"{start} --> {end}")
            lines.append(seg["text"].strip())
            lines.append("")
<<<<<<< HEAD
        
        return "\n".join(lines)
    
    @staticmethod
    def _format_timestamp(seconds: float) -> str:
        """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
=======

        return "\n".join(lines)

    @staticmethod
    def _format_timestamp(seconds: float) -> str:
>>>>>>> 243afb8 (init commit  in github)
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        ms = int((seconds % 1) * 1000)
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


class BaseTranscriber(ABC):
<<<<<<< HEAD
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
=======
    @abstractmethod
    def transcribe(
            self,
            audio_path: Path,
            language: str | None = None
    ) -> TranscriptionResult:
        pass

    @abstractmethod
    def transcribe_batch(
            self,
            audio_paths: list[Path],
            language: str | None = None
    ) -> Iterator[tuple[Path, TranscriptionResult]]:
>>>>>>> 243afb8 (init commit  in github)
        pass


class WhisperTranscriber(BaseTranscriber):
<<<<<<< HEAD
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
        
=======
    """Local Whisper transcription."""

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
        # Конвертируем в wav если нужно (whisper любит wav/mp3)
        audio_path = self._ensure_compatible(audio_path)

        options = {}
        if language:
            options["language"] = language

        result = self.model.transcribe(str(audio_path), **options)

>>>>>>> 243afb8 (init commit  in github)
        return TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", "unknown"),
            duration=result["segments"][-1]["end"] if result["segments"] else 0,
            segments=result["segments"],
        )
<<<<<<< HEAD
    
    def transcribe_batch(
        self, 
        audio_paths: list[Path],
        language: str | None = None
    ) -> Iterator[tuple[Path, TranscriptionResult]]:
        """Transcribe multiple audio files."""
=======

    def transcribe_batch(
            self,
            audio_paths: list[Path],
            language: str | None = None
    ) -> Iterator[tuple[Path, TranscriptionResult]]:
>>>>>>> 243afb8 (init commit  in github)
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
<<<<<<< HEAD
    
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
        
=======

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

        # Temp file
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        audio.export(tmp.name, format="wav")

>>>>>>> 243afb8 (init commit  in github)
        return Path(tmp.name)


class FasterWhisperTranscriber(BaseTranscriber):
    """Faster Whisper using CTranslate2 - 4x faster on CPU."""
<<<<<<< HEAD
    
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
        
=======

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

>>>>>>> 243afb8 (init commit  in github)
        self.model = WhisperModel(
            model_size.value,
            device=device,
            compute_type=compute_type
        )
<<<<<<< HEAD
    
    def transcribe(
        self, 
        audio_path: Path, 
        language: str | None = None
    ) -> TranscriptionResult:
        """Transcribe a single audio file."""
=======

    def transcribe(
            self,
            audio_path: Path,
            language: str | None = None
    ) -> TranscriptionResult:
>>>>>>> 243afb8 (init commit  in github)
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
<<<<<<< HEAD
            vad_filter=True,
        )
        
        segments_list = []
        full_text = []
        
=======
            vad_filter=True,  # убирает тишину
        )

        segments_list = []
        full_text = []

>>>>>>> 243afb8 (init commit  in github)
        for seg in segments:
            segments_list.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
            })
            full_text.append(seg.text)
<<<<<<< HEAD
        
=======

>>>>>>> 243afb8 (init commit  in github)
        return TranscriptionResult(
            text=" ".join(full_text).strip(),
            language=info.language,
            duration=info.duration,
            segments=segments_list,
            confidence=info.language_probability,
        )
<<<<<<< HEAD
    
    def transcribe_batch(
        self, 
        audio_paths: list[Path],
        language: str | None = None
    ) -> Iterator[tuple[Path, TranscriptionResult]]:
        """Transcribe multiple audio files."""
=======

    def transcribe_batch(
            self,
            audio_paths: list[Path],
            language: str | None = None
    ) -> Iterator[tuple[Path, TranscriptionResult]]:
>>>>>>> 243afb8 (init commit  in github)
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
<<<<<<< HEAD
    
    def __init__(self, api_key: str | None = None):
        from openai import OpenAI
        
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
    
    def transcribe(
        self, 
        audio_path: Path, 
        language: str | None = None
    ) -> TranscriptionResult:
        """Transcribe a single audio file via API."""
=======

    def __init__(self, api_key: str | None = None):
        from openai import OpenAI

        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def transcribe(
            self,
            audio_path: Path,
            language: str | None = None
    ) -> TranscriptionResult:
>>>>>>> 243afb8 (init commit  in github)
        with open(audio_path, "rb") as f:
            kwargs = {"model": "whisper-1", "file": f}
            if language:
                kwargs["language"] = language
<<<<<<< HEAD
            
            kwargs["response_format"] = "verbose_json"
            
            response = self.client.audio.transcriptions.create(**kwargs)
        
=======

            # Verbose JSON для timestamps
            kwargs["response_format"] = "verbose_json"

            response = self.client.audio.transcriptions.create(**kwargs)

>>>>>>> 243afb8 (init commit  in github)
        return TranscriptionResult(
            text=response.text,
            language=response.language,
            duration=response.duration,
            segments=[
                {"start": s.start, "end": s.end, "text": s.text}
                for s in (response.segments or [])
            ]
        )
<<<<<<< HEAD
    
    def transcribe_batch(
        self, 
        audio_paths: list[Path],
        language: str | None = None
    ) -> Iterator[tuple[Path, TranscriptionResult]]:
        """Transcribe multiple audio files via API."""
=======

    def transcribe_batch(
            self,
            audio_paths: list[Path],
            language: str | None = None
    ) -> Iterator[tuple[Path, TranscriptionResult]]:
>>>>>>> 243afb8 (init commit  in github)
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
<<<<<<< HEAD
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
=======
        backend: Backend = Backend.FASTER_WHISPER,
        model_size: ModelSize = ModelSize.BASE,
        **kwargs
) -> BaseTranscriber:
    """Factory для создания транскрайбера."""

>>>>>>> 243afb8 (init commit  in github)
    if backend == Backend.WHISPER:
        return WhisperTranscriber(model_size, **kwargs)
    elif backend == Backend.FASTER_WHISPER:
        return FasterWhisperTranscriber(model_size, **kwargs)
    elif backend == Backend.OPENAI_API:
        return OpenAIAPITranscriber(**kwargs)
    else:
<<<<<<< HEAD
        raise ValueError(f"Unknown backend: {backend}")
=======
        raise ValueError(f"Unknown backend: {backend}")
>>>>>>> 243afb8 (init commit  in github)
