#!/usr/bin/env python3
"""
Example Telegram Bot with Voice-to-Text
========================================

A complete example bot using aiogram 3.x that:
- Transcribes voice messages
- Transcribes video notes (round videos)
- Transcribes audio files
- Supports multiple languages

Usage:
    export TELEGRAM_BOT_TOKEN="your-token-here"
    python examples/telegram_bot.py

Or with custom settings:
    WHISPER_MODEL=small WHISPER_LANGUAGE=ru python examples/telegram_bot.py
"""

import asyncio
import logging
import os
import sys

# Add parent to path for local development
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import Command
from aiogram.enums import ParseMode

from voice_to_text.integrations.telegram import TelegramVoiceHandler
from voice_to_text.transcriber import Backend, ModelSize

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Configuration from environment
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE")  # None = auto-detect
WHISPER_BACKEND = os.getenv("WHISPER_BACKEND", "faster")

# Model size mapping
MODEL_MAP = {
    "tiny": ModelSize.TINY,
    "base": ModelSize.BASE,
    "small": ModelSize.SMALL,
    "medium": ModelSize.MEDIUM,
    "large": ModelSize.LARGE,
}

# Backend mapping
BACKEND_MAP = {
    "whisper": Backend.WHISPER,
    "faster": Backend.FASTER_WHISPER,
    "api": Backend.OPENAI_API,
}


class VoiceToTextBot:
    """Telegram bot with voice transcription."""
    
    def __init__(self, token: str):
        self.bot = Bot(token=token, parse_mode=ParseMode.HTML)
        self.dp = Dispatcher()
        
        # Initialize voice handler
        self.voice_handler = TelegramVoiceHandler(
            backend=BACKEND_MAP.get(WHISPER_BACKEND, Backend.FASTER_WHISPER),
            model_size=MODEL_MAP.get(WHISPER_MODEL, ModelSize.BASE),
            language=WHISPER_LANGUAGE,
        )
        
        # Register handlers
        self._register_handlers()
        
        logger.info(f"Bot initialized with model={WHISPER_MODEL}, lang={WHISPER_LANGUAGE or 'auto'}")
    
    def _register_handlers(self):
        """Register message handlers."""
        
        @self.dp.message(Command("start"))
        async def cmd_start(message: Message):
            await message.answer(
                "👋 <b>Voice-to-Text Bot</b>\n\n"
                "Отправь мне голосовое сообщение, кружочек или аудиофайл — "
                "я расшифрую его в текст!\n\n"
                "🎤 Голосовые — /voice\n"
                "🔵 Кружочки — /video_note\n"
                "🎵 Аудиофайлы — /audio\n\n"
                f"⚙️ Модель: <code>{WHISPER_MODEL}</code>\n"
                f"🌐 Язык: <code>{WHISPER_LANGUAGE or 'авто'}</code>"
            )
        
        @self.dp.message(Command("help"))
        async def cmd_help(message: Message):
            await message.answer(
                "📖 <b>Как пользоваться:</b>\n\n"
                "1. Отправь голосовое сообщение\n"
                "2. Подожди пару секунд\n"
                "3. Получи текст!\n\n"
                "💡 <b>Команды:</b>\n"
                "/start — начать\n"
                "/help — эта справка\n"
                "/lang &lt;code&gt; — сменить язык (ru, en, ...)\n"
                "/model — текущая модель\n\n"
                "🔧 <b>Поддерживаемые форматы:</b>\n"
                "• Голосовые сообщения (.ogg)\n"
                "• Кружочки (video notes)\n"
                "• Аудиофайлы (.mp3, .wav, .m4a, ...)"
            )
        
        @self.dp.message(Command("model"))
        async def cmd_model(message: Message):
            await message.answer(
                f"⚙️ <b>Текущие настройки:</b>\n\n"
                f"Модель: <code>{WHISPER_MODEL}</code>\n"
                f"Бэкенд: <code>{WHISPER_BACKEND}</code>\n"
                f"Язык: <code>{WHISPER_LANGUAGE or 'авто-определение'}</code>"
            )
        
        @self.dp.message(F.voice)
        async def handle_voice(message: Message):
            """Handle voice messages."""
            processing_msg = await message.answer("🎤 Расшифровываю...")
            
            try:
                result = await self.voice_handler.transcribe_voice(
                    message, self.bot
                )
                
                await processing_msg.delete()
                await message.reply(
                    f"📝 <b>Текст:</b>\n\n{result.text}\n\n"
                    f"<i>🌐 {result.language.upper()} · "
                    f"⏱ {result.result.duration:.1f}с</i>"
                )
                
                logger.info(
                    f"Transcribed voice from {message.from_user.id}: "
                    f"{len(result.text)} chars, {result.result.duration:.1f}s"
                )
                
            except Exception as e:
                await processing_msg.delete()
                await message.reply(f"❌ Ошибка: {e}")
                logger.error(f"Voice transcription error: {e}")
        
        @self.dp.message(F.video_note)
        async def handle_video_note(message: Message):
            """Handle video notes (round videos)."""
            processing_msg = await message.answer("🔵 Расшифровываю кружочек...")
            
            try:
                result = await self.voice_handler.transcribe_video_note(
                    message, self.bot
                )
                
                await processing_msg.delete()
                await message.reply(
                    f"📝 <b>Текст:</b>\n\n{result.text}\n\n"
                    f"<i>🌐 {result.language.upper()} · "
                    f"⏱ {result.result.duration:.1f}с</i>"
                )
                
                logger.info(
                    f"Transcribed video_note from {message.from_user.id}: "
                    f"{len(result.text)} chars"
                )
                
            except Exception as e:
                await processing_msg.delete()
                await message.reply(f"❌ Ошибка: {e}")
                logger.error(f"Video note transcription error: {e}")
        
        @self.dp.message(F.audio)
        async def handle_audio(message: Message):
            """Handle audio files."""
            audio = message.audio
            
            # Check file size (Telegram limit for bots is 20MB)
            if audio.file_size and audio.file_size > 20 * 1024 * 1024:
                await message.reply("❌ Файл слишком большой (макс. 20 МБ)")
                return
            
            processing_msg = await message.answer(
                f"🎵 Расшифровываю <code>{audio.file_name or 'audio'}</code>..."
            )
            
            try:
                result = await self.voice_handler.transcribe_audio(
                    message, self.bot
                )
                
                await processing_msg.delete()
                await message.reply(
                    f"📝 <b>Текст:</b>\n\n{result.text}\n\n"
                    f"<i>🌐 {result.language.upper()} · "
                    f"⏱ {result.result.duration:.1f}с</i>"
                )
                
                logger.info(
                    f"Transcribed audio from {message.from_user.id}: "
                    f"{len(result.text)} chars, {result.result.duration:.1f}s"
                )
                
            except Exception as e:
                await processing_msg.delete()
                await message.reply(f"❌ Ошибка: {e}")
                logger.error(f"Audio transcription error: {e}")
    
    async def run(self):
        """Start the bot."""
        logger.info("Starting bot...")
        
        # Pre-load model
        logger.info("Loading Whisper model...")
        await self.voice_handler.get_transcriber()
        logger.info("Model loaded!")
        
        # Start polling
        await self.dp.start_polling(self.bot)


async def main():
    if not BOT_TOKEN:
        print("Error: TELEGRAM_BOT_TOKEN not set")
        print("Usage: TELEGRAM_BOT_TOKEN=your-token python examples/telegram_bot.py")
        sys.exit(1)
    
    bot = VoiceToTextBot(BOT_TOKEN)
    await bot.run()


if __name__ == "__main__":
    asyncio.run(main())
