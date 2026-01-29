# Telegram Integration

Voice-to-Text integration for Telegram bots. Supports both **aiogram 3.x** and **python-telegram-bot**.

## Installation

```bash
# Base installation
pip install -r requirements.txt

# For aiogram
pip install aiogram>=3.0

# For python-telegram-bot
pip install python-telegram-bot>=20.0
```

## Quick Start

### aiogram 3.x

```python
from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from voice_to_text.integrations.telegram import TelegramVoiceHandler

# Initialize
bot = Bot(token="YOUR_TOKEN")
dp = Dispatcher()
handler = TelegramVoiceHandler(language="ru")

@dp.message(F.voice)
async def handle_voice(message: Message):
    result = await handler.transcribe_voice(message, bot)
    await message.reply(result.text)

# Run
dp.run_polling(bot)
```

### python-telegram-bot

```python
from telegram import Update
from telegram.ext import Application, MessageHandler, filters
from voice_to_text.integrations.telegram import TelegramVoiceHandler

handler = TelegramVoiceHandler(language="ru")

async def voice_handler(update: Update, context):
    result = await handler.transcribe_from_ptb(update.message, context.bot)
    if result:
        await update.message.reply_text(result.text)

app = Application.builder().token("YOUR_TOKEN").build()
app.add_handler(MessageHandler(filters.VOICE, voice_handler))
app.run_polling()
```

## Features

### Supported Message Types

| Type | Method | Description |
|------|--------|-------------|
| Voice | `transcribe_voice()` | Standard voice messages (.ogg) |
| Video Note | `transcribe_video_note()` | Round video messages |
| Audio | `transcribe_audio()` | Audio file attachments |
| Auto | `transcribe_message()` | Auto-detect type |

### Configuration

```python
from voice_to_text.integrations.telegram import TelegramVoiceHandler
from voice_to_text.transcriber import Backend, ModelSize

handler = TelegramVoiceHandler(
    # Transcription backend
    backend=Backend.FASTER_WHISPER,  # or WHISPER, OPENAI_API
    
    # Model size (for local backends)
    model_size=ModelSize.SMALL,  # tiny, base, small, medium, large
    
    # Default language (None = auto-detect)
    language="ru",
    
    # Temp files directory
    temp_dir="/tmp/voice_to_text",
    
    # Auto-cleanup temp files
    cleanup=True,
)
```

### Result Object

```python
result = await handler.transcribe_voice(message, bot)

# Access fields
result.text          # Transcribed text
result.language      # Detected language
result.message_type  # TelegramMessageType.VOICE
result.file_id       # Telegram file_id
result.duration      # Duration from Telegram metadata
result.file_size     # File size in bytes

# Underlying TranscriptionResult
result.result.segments   # Word-level timestamps
result.result.to_srt()   # Export as SRT subtitles

# Formatted reply
text = result.format_reply(include_stats=True)
# "Transcribed text\n\n🌐 RU · ⏱ 0:15"
```

### Convenience Function

```python
from voice_to_text.integrations.telegram import transcribe_telegram_voice

# Quick one-liner
result = await transcribe_telegram_voice(
    message.voice.file_id,
    bot,
    language="ru"
)
print(result.text)
```

## Example Bot

See [examples/telegram_bot.py](../examples/telegram_bot.py) for a complete bot implementation.

```bash
# Run example bot
export TELEGRAM_BOT_TOKEN="your-token"
export WHISPER_MODEL="small"     # tiny, base, small, medium, large
export WHISPER_LANGUAGE="ru"     # or leave empty for auto-detect

python examples/telegram_bot.py
```

## Integration with audiotools.dev Ecosystem

This module is designed to work with the unified Telegram bot for audiotools.dev:

```python
from voice_to_text.integrations.telegram import TelegramVoiceHandler
from music_recognition import MusicRecognizer  # from music-recognition
from audiobook_cleaner import AudioCleaner     # from audiobook-cleaner

class UnifiedAudioBot:
    def __init__(self):
        self.transcriber = TelegramVoiceHandler()
        self.music_recognizer = MusicRecognizer()
        self.audio_cleaner = AudioCleaner()
    
    async def handle_voice(self, message, bot):
        # Transcribe
        result = await self.transcriber.transcribe_voice(message, bot)
        return result.text
    
    async def handle_music(self, message, bot):
        # Recognize song
        ...
```

## Error Handling

```python
from voice_to_text.integrations.telegram import TelegramVoiceHandler

handler = TelegramVoiceHandler()

@dp.message(F.voice)
async def handle_voice(message: Message):
    try:
        result = await handler.transcribe_voice(message, bot)
        await message.reply(result.text)
        
    except ValueError as e:
        # No voice attachment
        await message.reply("❌ No voice message found")
        
    except RuntimeError as e:
        # Transcription failed
        await message.reply(f"❌ Transcription error: {e}")
        
    except Exception as e:
        # Other errors (network, etc.)
        await message.reply("❌ Something went wrong")
        logger.error(f"Voice error: {e}")
```

## Performance Tips

1. **Pre-load model** at startup:
   ```python
   # In your startup code
   await handler.get_transcriber()
   ```

2. **Use appropriate model size**:
   - `tiny`/`base` — fast, good for short messages
   - `small` — balanced, recommended for Russian
   - `medium`/`large` — best quality, slower

3. **Set language explicitly** for faster processing:
   ```python
   handler = TelegramVoiceHandler(language="ru")
   ```

4. **Use faster-whisper backend** (default) for 4x speedup on CPU.
