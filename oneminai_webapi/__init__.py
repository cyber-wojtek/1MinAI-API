"""
oneminai — Async Python client for the 1min.AI API.

Quick start::

    import asyncio
    import oneminai

    async def main():
        async with oneminai.OneMinAIClient("YOUR_API_KEY") as client:
            response = await client.generate_content("Hello!")
            print(response.text)

    asyncio.run(main())
"""

from .client import OneMinAIClient
from .constants import (
    AssetType,
    ChatModel,
    CodeModel,
    ImageModel,
    STTModel,
    TTSModel,
    TTSVoice,
    VideoModel,
)
from .exceptions import (
    APIError,
    AssetUploadError,
    AuthenticationError,
    ConversationError,
    OneMinAIError,
    RateLimitError,
    TimeoutError,
    ValidationError,
)
from .session import ChatSession
from .types import (
    AssetRecord,
    AudioOutput,
    ChatOutput,
    ConversationRecord,
    CreditEstimate,
    GeneratedImage,
    ImageOutput,
    MessageRecord,
    MusicOutput,
    NotificationRecord,
    TranscriptionOutput,
    UserRecord,
    VideoOutput,
)

import logging as _logging


def set_log_level(level: str) -> None:
    """
    Configure the ``1minai`` logger.

    Calling this removes any existing handlers on the ``1minai`` logger so the
    new level takes effect cleanly.

    Parameters
    ----------
    level:
        One of ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``,
        ``"CRITICAL"``.

    Example::

        from 1minai import set_log_level
        set_log_level("DEBUG")
    """
    log = _logging.getLogger("1minai")
    log.handlers.clear()
    handler = _logging.StreamHandler()
    handler.setFormatter(
        _logging.Formatter("%(asctime)s  %(levelname)-5s  %(name)s  %(message)s")
    )
    log.addHandler(handler)
    log.setLevel(getattr(_logging, level.upper(), _logging.INFO))


__all__ = [
    # Client
    "OneMinAIClient",
    # Session
    "ChatSession",
    # Models / enums
    "ChatModel",
    "CodeModel",
    "ImageModel",
    "TTSModel",
    "TTSVoice",
    "STTModel",
    "VideoModel",
    "AssetType",
    # Output types
    "ChatOutput",
    "CreditEstimate",
    "ImageOutput",
    "AudioOutput",
    "MusicOutput",
    "TranscriptionOutput",
    "VideoOutput",
    "AssetRecord",
    "ConversationRecord",
    "GeneratedImage",
    "MessageRecord",
    "NotificationRecord",
    "UserRecord",
    # Exceptions
    "OneMinAIError",
    "APIError",
    "AuthenticationError",
    "RateLimitError",
    "AssetUploadError",
    "ConversationError",
    "ValidationError",
    "TimeoutError",
    # Logging
    "set_log_level",
]

__version__ = "1.0.0"
