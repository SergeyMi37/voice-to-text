"""Audio format detection and conversion utilities."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import BinaryIO
import struct
import tempfile


class AudioFormat(Enum):
    """Supported audio formats."""
    OGG_OPUS = "ogg/opus"    # Telegram voice
    OGG_VORBIS = "ogg/vorbis"
    MP3 = "mp3"
    WAV = "wav"
    M4A = "m4a"              # WhatsApp voice
    AAC = "aac"
    WEBM = "webm"            # Web recorder
    AMR = "amr"              # Old phone recordings
    FLAC = "flac"
    UNKNOWN = "unknown"


@dataclass
class AudioInfo:
    """Audio file information."""
    format: AudioFormat
    duration: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    bitrate: int | None = None

    @property
    def is_voice_message(self) -> bool:
        """Check if format is typical for voice messages."""
        return self.format in {
            AudioFormat.OGG_OPUS,   # Telegram
            AudioFormat.M4A,        # WhatsApp
            AudioFormat.AMR,        # SMS/MMS
            AudioFormat.WEBM,       # Web
        }


# Magic bytes for format detection
MAGIC_BYTES = {
    b"OggS": AudioFormat.OGG_OPUS,  # Will check further for opus/vorbis
    b"ID3": AudioFormat.MP3,
    b"\xff\xfb": AudioFormat.MP3,
    b"\xff\xfa": AudioFormat.MP3,
    b"\xff\xf3": AudioFormat.MP3,
    b"\xff\xf2": AudioFormat.MP3,
    b"RIFF": AudioFormat.WAV,
    b"fLaC": AudioFormat.FLAC,
    b"\x1aE\xdf\xa3": AudioFormat.WEBM,  # EBML header (webm/mkv)
    b"#!AMR": AudioFormat.AMR,
}


def detect_format(file_path: Path) -> AudioFormat:
    """Detect audio format by magic bytes and extension."""
    with open(file_path, "rb") as f:
        header = f.read(32)

    # Check magic bytes
    for magic, fmt in MAGIC_BYTES.items():
        if header.startswith(magic):
            # Special case: distinguish OGG Opus from Vorbis
            if fmt == AudioFormat.OGG_OPUS:
                return _detect_ogg_codec(header)
            return fmt

    # Check for M4A/AAC (ftyp box)
    if b"ftyp" in header[:12]:
        if b"M4A" in header or b"mp42" in header or b"isom" in header:
            return AudioFormat.M4A
        return AudioFormat.AAC

    # Fallback to extension
    return _format_from_extension(file_path)


def _detect_ogg_codec(header: bytes) -> AudioFormat:
    """Detect codec inside OGG container."""
    # OpusHead signature appears after OGG page header
    if b"OpusHead" in header:
        return AudioFormat.OGG_OPUS
    if b"vorbis" in header:
        return AudioFormat.OGG_VORBIS
    return AudioFormat.OGG_OPUS  # Default assumption for voice


def _format_from_extension(path: Path) -> AudioFormat:
    """Fallback format detection by extension."""
    ext = path.suffix.lower()
    mapping = {
        ".ogg": AudioFormat.OGG_OPUS,
        ".opus": AudioFormat.OGG_OPUS,
        ".mp3": AudioFormat.MP3,
        ".wav": AudioFormat.WAV,
        ".m4a": AudioFormat.M4A,
        ".aac": AudioFormat.AAC,
        ".webm": AudioFormat.WEBM,
        ".amr": AudioFormat.AMR,
        ".flac": AudioFormat.FLAC,
    }
    return mapping.get(ext, AudioFormat.UNKNOWN)


def get_audio_info(file_path: Path) -> AudioInfo:
    """Get detailed audio information."""
    fmt = detect_format(file_path)
    info = AudioInfo(format=fmt)

    try:
        # Use pydub for detailed info (requires ffmpeg for some formats)
        from pydub import AudioSegment
        from pydub.utils import mediainfo

        media_info = mediainfo(str(file_path))

        info.duration = float(media_info.get("duration", 0))
        info.sample_rate = int(media_info.get("sample_rate", 0))
        info.channels = int(media_info.get("channels", 0))
        info.bitrate = int(media_info.get("bit_rate", 0))

    except Exception:
        # Fallback: try to get duration from file size estimate
        pass

    return info


def convert_to_wav(
    input_path: Path,
    output_path: Path | None = None,
    sample_rate: int = 16000,
    channels: int = 1
) -> Path:
    """Convert audio to WAV format optimized for speech recognition.

    Whisper works best with:
    - 16kHz sample rate
    - Mono channel
    - 16-bit PCM
    """
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(input_path))

    # Resample and convert to mono
    audio = audio.set_frame_rate(sample_rate)
    audio = audio.set_channels(channels)

    # Output path
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        output_path = Path(tmp.name)

    audio.export(str(output_path), format="wav")

    return output_path


def convert_to_mp3(
    input_path: Path,
    output_path: Path | None = None,
    bitrate: str = "128k"
) -> Path:
    """Convert audio to MP3."""
    from pydub import AudioSegment

    audio = AudioSegment.from_file(str(input_path))

    if output_path is None:
        output_path = input_path.with_suffix(".mp3")

    audio.export(str(output_path), format="mp3", bitrate=bitrate)

    return output_path


def is_supported(file_path: Path) -> bool:
    """Check if file format is supported for transcription."""
    fmt = detect_format(file_path)
    return fmt != AudioFormat.UNKNOWN


def get_supported_extensions() -> set[str]:
    """Get set of supported file extensions."""
    return {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".aac", ".webm", ".amr", ".flac"}


def find_voice_files(directory: Path, recursive: bool = False) -> list[Path]:
    """Find all voice/audio files in directory."""
    extensions = get_supported_extensions()

    if recursive:
        files = []
        for ext in extensions:
            files.extend(directory.rglob(f"*{ext}"))
        return sorted(files)
    else:
        return sorted([
            f for f in directory.iterdir()
            if f.suffix.lower() in extensions
        ])


@dataclass
class ConversionResult:
    """Result of audio conversion."""
    success: bool
    input_path: Path
    output_path: Path | None
    input_format: AudioFormat
    output_format: AudioFormat
    error: str | None = None


def batch_convert(
    input_files: list[Path],
    output_dir: Path,
    target_format: str = "wav",
    **kwargs
) -> list[ConversionResult]:
    """Batch convert multiple audio files."""
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for input_path in input_files:
        input_fmt = detect_format(input_path)
        output_path = output_dir / f"{input_path.stem}.{target_format}"

        try:
            if target_format == "wav":
                convert_to_wav(input_path, output_path, **kwargs)
            elif target_format == "mp3":
                convert_to_mp3(input_path, output_path, **kwargs)
            else:
                raise ValueError(f"Unsupported target format: {target_format}")

            results.append(ConversionResult(
                success=True,
                input_path=input_path,
                output_path=output_path,
                input_format=input_fmt,
                output_format=AudioFormat(target_format) if target_format in ["mp3", "wav"] else AudioFormat.UNKNOWN,
            ))

        except Exception as e:
            results.append(ConversionResult(
                success=False,
                input_path=input_path,
                output_path=None,
                input_format=input_fmt,
                output_format=AudioFormat.UNKNOWN,
                error=str(e),
            ))

    return results