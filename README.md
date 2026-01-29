# voice-to-text

<<<<<<< HEAD

Bulk voice message transcription using Whisper</strong>


=======
>>>>>>> 243afb8 (init commit  in github)
---

Convert voice messages from Telegram, WhatsApp, and other messengers to text. Batch processing, multiple output formats, three transcription backends.

## Features

- 🚀 **3 backends** — local Whisper, faster-whisper (4x speedup), OpenAI API
- 📱 **Messenger-friendly** — native support for Telegram (.ogg/.opus) and WhatsApp (.m4a)
- 📁 **Batch processing** — transcribe entire folders of voice messages
- 📝 **Multiple formats** — text, JSON, SRT subtitles
- 🌍 **Language detection** — automatic or manual language selection
- ⚡ **GPU acceleration** — CUDA support for faster processing

## Installation

### Basic (CPU)
<<<<<<< HEAD

=======
>>>>>>> 243afb8 (init commit  in github)
```bash
pip install -r requirements.txt
```

### With faster-whisper (recommended)
<<<<<<< HEAD

=======
>>>>>>> 243afb8 (init commit  in github)
```bash
pip install -r requirements.txt
pip install faster-whisper
```

### With GPU support
<<<<<<< HEAD

=======
>>>>>>> 243afb8 (init commit  in github)
```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### System dependencies

For `.ogg`, `.opus`, `.m4a` conversion you need FFmpeg:
<<<<<<< HEAD

=======
>>>>>>> 243afb8 (init commit  in github)
```bash
# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg

# Windows
choco install ffmpeg
```

## Usage

### Command Line
<<<<<<< HEAD

```bash
# Single file
vtt transcribe voice.ogg

# With output file
vtt transcribe voice.ogg -o transcript.txt

# Batch processing (entire folder)
vtt transcribe ./voices --batch -o results.json -f json

# Specify language (faster, more accurate)
vtt transcribe voice.ogg -l ru

# Use larger model for better quality
vtt transcribe voice.ogg -m medium

# Generate SRT subtitles
vtt transcribe podcast.mp3 -f srt -o podcast.srt

# Use OpenAI API (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
vtt transcribe voice.ogg -b api

# List available models
vtt models
```

### Python API

=======
```bash
# Single file
python -m voice_to_text.cli transcribe voice.ogg

# With output file
python -m voice_to_text.cli transcribe voice.ogg -o transcript.txt

# Batch processing (entire folder)
python -m voice_to_text.cli transcribe ./voices --batch -o results.json -f json

# Specify language (faster, more accurate)
python -m voice_to_text.cli transcribe voice.ogg -l ru

# Use larger model for better quality
python -m voice_to_text.cli transcribe voice.ogg -m medium

# Generate SRT subtitles
python -m voice_to_text.cli transcribe podcast.mp3 -f srt -o podcast.srt

# Use OpenAI API (requires OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python -m voice_to_text.cli transcribe voice.ogg -b api

# List available models
python -m voice_to_text.cli models
```

### Python API
>>>>>>> 243afb8 (init commit  in github)
```python
from pathlib import Path
from voice_to_text.transcriber import create_transcriber, Backend, ModelSize

# Create transcriber
transcriber = create_transcriber(
    backend=Backend.FASTER_WHISPER,
    model_size=ModelSize.BASE
)

# Single file
result = transcriber.transcribe(Path("voice.ogg"), language="ru")
print(result.text)
print(f"Duration: {result.duration}s, Language: {result.language}")

# Batch processing
voices = list(Path("./voices").glob("*.ogg"))
for path, result in transcriber.transcribe_batch(voices):
    print(f"{path.name}: {result.text[:50]}...")

# Export as SRT
srt_content = result.to_srt()
Path("output.srt").write_text(srt_content)
```

### Format Detection
<<<<<<< HEAD

=======
>>>>>>> 243afb8 (init commit  in github)
```python
from voice_to_text.formats import detect_format, get_audio_info, find_voice_files

# Detect format
fmt = detect_format(Path("voice.ogg"))
print(fmt)  # AudioFormat.OGG_OPUS

# Get detailed info
info = get_audio_info(Path("voice.ogg"))
print(f"Duration: {info.duration}s, Sample rate: {info.sample_rate}Hz")

# Find all voice files
files = find_voice_files(Path("./downloads"), recursive=True)
```

## Backends

| Backend | Speed | Quality | GPU | Offline | Cost |
|---------|-------|---------|-----|---------|------|
| `whisper` | 1x | ★★★★★ | ✅ | ✅ | Free |
| `faster` | 4x | ★★★★★ | ✅ | ✅ | Free |
| `api` | ~2x | ★★★★★ | N/A | ❌ | $0.006/min |

### Choosing a backend

- **faster** (default) — best for most cases, 4x faster than original Whisper on CPU
- **whisper** — original OpenAI Whisper, use if you have issues with faster-whisper
- **api** — no local compute needed, pay-per-use, requires internet

## Models

| Model | Parameters | VRAM | Relative Speed | Quality |
|-------|------------|------|----------------|---------|
| tiny | 39M | ~1 GB | ~32x | ★☆☆☆☆ |
| base | 74M | ~1 GB | ~16x | ★★☆☆☆ |
| small | 244M | ~2 GB | ~6x | ★★★☆☆ |
| medium | 769M | ~5 GB | ~2x | ★★★★☆ |
| large-v3 | 1550M | ~10 GB | 1x | ★★★★★ |

**Recommendations:**
- Short voice messages (< 1 min): `base` or `small`
- Long recordings, podcasts: `medium` or `large-v3`
- Non-English languages: `medium` or `large-v3`
- Low-end hardware: `tiny` or `base`

## Supported Formats

| Format | Extension | Source |
|--------|-----------|--------|
| OGG Opus | `.ogg`, `.opus` | Telegram |
| M4A/AAC | `.m4a`, `.aac` | WhatsApp, iOS |
| MP3 | `.mp3` | Universal |
| WAV | `.wav` | Universal |
| WebM | `.webm` | Web recorders |
| AMR | `.amr` | Old phones, MMS |
| FLAC | `.flac` | Lossless |

## Examples

### Transcribe Telegram voice folder
<<<<<<< HEAD

=======
>>>>>>> 243afb8 (init commit  in github)
```bash
# Export voices from Telegram Desktop: 
# Settings → Advanced → Export Telegram Data → Voice Messages

<<<<<<< HEAD
vtt transcribe ~/Downloads/Telegram\ Desktop/voice_messages \
=======
python -m voice_to_text.cli transcribe ~/Downloads/Telegram\ Desktop/voice_messages \
>>>>>>> 243afb8 (init commit  in github)
    --batch \
    -l ru \
    -m small \
    -o telegram_voices.json \
    -f json
```

### Create podcast transcript with timestamps
<<<<<<< HEAD

```bash
vtt transcribe podcast.mp3 -m medium -f srt -o podcast.srt
```

### Process with Python script

=======
```bash
python -m voice_to_text.cli transcribe podcast.mp3 -m medium -f srt -o podcast.srt
```

### Process with Python script
>>>>>>> 243afb8 (init commit  in github)
```python
import json
from pathlib import Path
from voice_to_text.transcriber import create_transcriber, Backend, ModelSize
from voice_to_text.formats import find_voice_files

# Setup
transcriber = create_transcriber(
    backend=Backend.FASTER_WHISPER,
    model_size=ModelSize.SMALL
)

# Find and process
voices_dir = Path("./telegram_voices")
files = find_voice_files(voices_dir)

results = {}
for path, result in transcriber.transcribe_batch(files, language="ru"):
    results[path.name] = {
        "text": result.text,
        "duration": result.duration,
        "language": result.language
    }

# Save
Path("transcripts.json").write_text(
    json.dumps(results, ensure_ascii=False, indent=2)
)
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | API key for OpenAI backend |
| `WHISPER_CACHE_DIR` | Custom cache directory for models |
| `CUDA_VISIBLE_DEVICES` | GPU selection for CUDA |

## Troubleshooting

### FFmpeg not found
<<<<<<< HEAD

=======
>>>>>>> 243afb8 (init commit  in github)
```bash
# Check installation
ffmpeg -version

# Install if missing
sudo apt install ffmpeg  # Linux
brew install ffmpeg      # macOS
```

### CUDA out of memory

Use smaller model or switch to CPU:
<<<<<<< HEAD

```bash
# Smaller model
vtt transcribe voice.ogg -m base

# Force CPU
CUDA_VISIBLE_DEVICES="" vtt transcribe voice.ogg
=======
```bash
# Smaller model
python -m voice_to_text.cli transcribe voice.ogg -m base

# Force CPU
CUDA_VISIBLE_DEVICES="" python -m voice_to_text.cli transcribe voice.ogg
>>>>>>> 243afb8 (init commit  in github)
```

### Slow on CPU

Use faster-whisper backend:
<<<<<<< HEAD

```bash
pip install faster-whisper
vtt transcribe voice.ogg -b faster
```

## Development

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run tests
pytest

# Run tests with coverage
pytest --cov=voice_to_text --cov-report=html

# Format code
black voice_to_text tests
ruff check voice_to_text tests
=======
```bash
pip install faster-whisper
python -m voice_to_text.cli transcribe voice.ogg -b faster
>>>>>>> 243afb8 (init commit  in github)
```

## License

MIT

## Links

- 🌐 [audiotools.dev](https://audiotools.dev)
<<<<<<< HEAD
- 📦 [go-audio-converter](https://github.com/roman/go-audio-converter)
- 🎵 [music-recognition](https://github.com/roman/music-recognition)
=======
- 📦 [go-audio-converter](https://github.com/raxod/go-audio-converter)
- 🎵 [music-recognition](https://github.com/raxod/music-recognition)
>>>>>>> 243afb8 (init commit  in github)

---

<p align="center">
  Made with ❤️ for voice message haters
<<<<<<< HEAD
</p>
=======
</p>
>>>>>>> 243afb8 (init commit  in github)
