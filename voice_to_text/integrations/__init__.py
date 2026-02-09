# voice_to_text/integrations/__init__.py
"""Integrations with external services."""


def __getattr__(name):
    if name == "TelegramVoiceHandler":
        from .telegram import TelegramVoiceHandler
        return TelegramVoiceHandler
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["TelegramVoiceHandler"]
