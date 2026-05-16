from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import aiohttp


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

async def _download(url: str, dest: Path) -> None:
    async with aiohttp.ClientSession() as s:
        async with s.get(url) as r:
            r.raise_for_status()
            dest.write_bytes(await r.read())


# ──────────────────────────────────────────────────────────────────────────────
# Chat
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ChatOutput:
    """
    Structured result of a chat or text-generation call.

    Attributes
    ----------
    text:
        Full accumulated response text.
    text_delta:
        In streaming mode, the new text received in this chunk only.
        Empty for non-streaming outputs.
    model:
        Model that produced the response.
    conversation_id:
        Server-assigned conversation UUID when history is enabled.
    record_id:
        UUID of this AI record on the server.
    metadata:
        Raw ``aiRecord`` dict from the API response.
    """

    text: str
    text_delta: str = ""
    model: str = ""
    conversation_id: str | None = None
    record_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        preview = self.text[:120].replace("\
", " ")
        return f"<ChatOutput model={self.model!r} text={preview!r}>"


# ──────────────────────────────────────────────────────────────────────────────
# Images
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class GeneratedImage:
    """A single image returned by any image API call."""

    url: str
    model: str = ""

    def __repr__(self) -> str:
        return f"<GeneratedImage model={self.model!r} url={self.url!r}>"

    async def save(
        self,
        path: str | Path = ".",
        filename: str | None = None,
        verbose: bool = False,
    ) -> Path:
        """Download and save the image.  Returns the absolute path."""
        dest_dir = Path(path)
        dest_dir.mkdir(parents=True, exist_ok=True)
        fname = filename or (self.url.split("/")[-1].split("?")[0] or "image.png")
        dest  = dest_dir / fname
        await _download(self.url, dest)
        if verbose:
            print(f"Saved: {dest.resolve()}")
        return dest.resolve()


@dataclass
class ImageOutput:
    """Result of any image API call (generate, edit, upscale, etc.)."""

    images: list[GeneratedImage] = field(default_factory=list)
    model: str = ""
    record_id: str | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def image(self) -> GeneratedImage | None:
        """Convenience accessor for the first image."""
        return self.images[0] if self.images else None

    def __repr__(self) -> str:
        return f"<ImageOutput model={self.model!r} images={len(self.images)}>"


# ──────────────────────────────────────────────────────────────────────────────
# Audio
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AudioOutput:
    """Result of a text-to-speech or audio-processing call."""

    audio_url: str = ""
    model: str = ""
    record_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<AudioOutput model={self.model!r} url={self.audio_url!r}>"

    async def save(
        self,
        path: str | Path = ".",
        filename: str | None = None,
        verbose: bool = False,
    ) -> Path:
        """Download and save the audio file.  Returns the absolute path."""
        dest_dir = Path(path)
        dest_dir.mkdir(parents=True, exist_ok=True)
        fname = filename or (self.audio_url.split("/")[-1].split("?")[0] or "audio.mp3")
        dest  = dest_dir / fname
        await _download(self.audio_url, dest)
        if verbose:
            print(f"Saved: {dest.resolve()}")
        return dest.resolve()


@dataclass
class TranscriptionOutput:
    """Result of a speech-to-text call."""

    text: str = ""
    model: str = ""
    record_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\
", " ")
        return f"<TranscriptionOutput model={self.model!r} text={preview!r}>"


# ──────────────────────────────────────────────────────────────────────────────
# Video
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class VideoOutput:
    """Result of a text-to-video or image-to-video call."""

    video_url: str = ""
    model: str = ""
    record_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<VideoOutput model={self.model!r} url={self.video_url!r}>"

    async def save(
        self,
        path: str | Path = ".",
        filename: str | None = None,
        verbose: bool = False,
    ) -> Path:
        """Download and save the video file.  Returns the absolute path."""
        dest_dir = Path(path)
        dest_dir.mkdir(parents=True, exist_ok=True)
        fname = filename or (self.video_url.split("/")[-1].split("?")[0] or "video.mp4")
        dest  = dest_dir / fname
        await _download(self.video_url, dest)
        if verbose:
            print(f"Saved: {dest.resolve()}")
        return dest.resolve()


# ──────────────────────────────────────────────────────────────────────────────
# Music
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class MusicOutput:
    """Result of a music-generation call."""

    audio_url: str = ""
    model: str = ""
    record_id: str | None = None
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return f"<MusicOutput model={self.model!r} url={self.audio_url!r}>"

    async def save(
        self,
        path: str | Path = ".",
        filename: str | None = None,
        verbose: bool = False,
    ) -> Path:
        dest_dir = Path(path)
        dest_dir.mkdir(parents=True, exist_ok=True)
        fname = filename or (self.audio_url.split("/")[-1].split("?")[0] or "music.mp3")
        dest  = dest_dir / fname
        await _download(self.audio_url, dest)
        if verbose:
            print(f"Saved: {dest.resolve()}")
        return dest.resolve()


# ──────────────────────────────────────────────────────────────────────────────
# Assets
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class AssetRecord:
    """
    Metadata for a file uploaded to the team Asset API.

    Attributes
    ----------
    asset_key:
        S3 path / key; use this to reference images in image-API calls.
    file_id:
        UUID from ``fileContent.uuid``; use this in ``files=[]`` chat attachments.
    asset_type:
        Asset category string (IMAGE, DOCUMENT, AUDIO, VIDEO).
    metadata:
        Raw API response dict (contains both ``asset`` and ``fileContent``).
    """

    asset_key: str = ""
    file_id: str = ""
    asset_type: str = ""
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"<AssetRecord type={self.asset_type!r} "
            f"key={self.asset_key!r} id={self.file_id!r}>"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Conversations
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class ConversationRecord:
    """A server-side conversation thread."""

    conversation_id: str = ""
    title: str = ""
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"<ConversationRecord id={self.conversation_id!r} "
            f"title={self.title!r}>"
        )


@dataclass
class MessageRecord:
    """
    A single message within a conversation's history.

    Attributes
    ----------
    role:
        ``"USER"`` or the model ID string for assistant turns.
    content:
        Full message text.
    record_id:
        UUID of the corresponding AI record.
    credit:
        Credits consumed by this message.
    execution_time:
        Seconds taken to produce the response.
    metadata:
        Raw message dict from the API.
    """

    role: str = ""
    content: str = ""
    record_id: str = ""
    credit: int = 0
    execution_time: float = 0.0
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        preview = self.content[:80].replace("\
", " ")
        return f"<MessageRecord role={self.role!r} content={preview!r}>"


# ──────────────────────────────────────────────────────────────────────────────
# Credits
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class CreditEstimate:
    """
    Per-model credit cost estimate returned by ``estimate_chat_cost``.

    Attributes
    ----------
    models:
        Dict mapping model ID → ``{"inputTokens": int, "estimatedCredit": int, ...}``.
    total_input_tokens:
        Sum of input tokens across all requested models.
    total_estimated_credit:
        Sum of estimated credits across all requested models.
    """

    models: dict = field(default_factory=dict)
    total_input_tokens: int = 0
    total_estimated_credit: int = 0

    def __repr__(self) -> str:
        return (
            f"<CreditEstimate models={list(self.models)} "
            f"total_credit={self.total_estimated_credit}>"
        )


# ──────────────────────────────────────────────────────────────────────────────
# User / team
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class UserRecord:
    """
    Current authenticated user.

    Attributes
    ----------
    user_id:
        UUID of the user.
    email:
        User email address.
    team_id:
        UUID of the first (primary) team.
    team_name:
        Display name of the primary team.
    credit:
        Current credit balance of the primary team.
    plan:
        Subscription plan string (e.g. ``"FREE"``, ``"PRO"``).
    settings:
        Raw user-settings dict.
    metadata:
        Full raw ``user`` object from the API.
    """

    user_id: str = ""
    email: str = ""
    team_id: str = ""
    team_name: str = ""
    credit: int = 0
    plan: str = ""
    settings: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"<UserRecord email={self.email!r} team_id={self.team_id!r} "
            f"credit={self.credit} plan={self.plan!r}>"
        )


# ──────────────────────────────────────────────────────────────────────────────
# Notifications
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class NotificationRecord:
    """A single notification from the server."""

    notification_id: str = ""
    type: str = ""
    scope: str = ""
    content: str = ""
    status: str = ""
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"<NotificationRecord id={self.notification_id!r} "
            f"type={self.type!r} status={self.status!r}>"
        )