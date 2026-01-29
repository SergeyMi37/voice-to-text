# voice_to_text/integrations/__init__.py
"""Integrations with external services."""

from .telegram import TelegramVoiceHandler

__all__ = ["TelegramVoiceHandler"]
