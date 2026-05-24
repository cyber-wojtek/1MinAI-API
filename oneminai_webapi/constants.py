"""
Constants and enumerations for 1minai.

All model IDs are verified against the live /models catalog captured in the
HAR session.  Pass any enum member **or** a raw string wherever a model is
accepted.
"""
from __future__ import annotations

from enum import Enum


# ──────────────────────────────────────────────────────────────────────────────
# Base URL
# ──────────────────────────────────────────────────────────────────────────────

BASE_URL: str = "https://api.1min.ai"

# X-App-Version observed in the HAR capture.
APP_VERSION: str = "1.1.45"

# ──────────────────────────────────────────────────────────────────────────────
# Chat / unified-chat models  (feature: UNIFY_CHAT_WITH_AI)
# ──────────────────────────────────────────────────────────────────────────────

class ChatModel(Enum):
    """
    Models available for ``UNIFY_CHAT_WITH_AI``.

    Any raw model-ID string that appears in the 1min.AI /models catalog can
    also be passed directly wherever a ``ChatModel`` is accepted.
    """

    # ── Alibaba Qwen ──────────────────────────────────────────────────────
    QWEN3_VL_PLUS   = "qwen3-vl-plus"
    QWEN3_VL_FLASH  = "qwen3-vl-flash"
    QWEN3_MAX       = "qwen3-max"
    QWEN_VL_PLUS    = "qwen-vl-plus"
    QWEN_VL_MAX     = "qwen-vl-max"
    QWEN_PLUS       = "qwen-plus"
    QWEN_MAX        = "qwen-max"
    QWEN_FLASH      = "qwen-flash"

    # ── Anthropic Claude ──────────────────────────────────────────────────
    CLAUDE_SONNET_4_6   = "claude-sonnet-4-6"
    CLAUDE_SONNET_4_5   = "claude-sonnet-4-5-20250929"
    CLAUDE_SONNET_4     = "claude-sonnet-4-20250514"
    CLAUDE_OPUS_4_6     = "claude-opus-4-6"
    CLAUDE_OPUS_4_5     = "claude-opus-4-5-20251101"
    CLAUDE_OPUS_4       = "claude-opus-4-20250514"
    CLAUDE_OPUS_4_1     = "claude-opus-4-1-20250805"
    CLAUDE_HAIKU_4_5    = "claude-haiku-4-5-20251001"

    # ── Google Gemini ─────────────────────────────────────────────────────
    GEMINI_3_1_PRO      = "gemini-3.1-pro-preview"
    GEMINI_3_1_FLASH    = "gemini-3.1-flash-lite-preview"
    GEMINI_3_FLASH      = "gemini-3-flash-preview"
    GEMINI_2_5_PRO      = "gemini-2.5-pro"
    GEMINI_2_5_FLASH    = "gemini-2.5-flash"

    # ── xAI Grok ──────────────────────────────────────────────────────────
    GROK_4              = "grok-4-0709"
    GROK_4_FAST         = "grok-4-fast-non-reasoning"
    GROK_4_REASONING    = "grok-4-fast-reasoning"
    GROK_3              = "grok-3"
    GROK_3_MINI         = "grok-3-mini"

    # ── OpenAI ────────────────────────────────────────────────────────────
    GPT_4O              = "gpt-4o"
    GPT_4O_MINI         = "gpt-4o-mini"
    GPT_4_1             = "gpt-4.1"
    GPT_4_1_MINI        = "gpt-4.1-mini"
    GPT_4_1_NANO        = "gpt-4.1-nano"
    GPT_4_TURBO         = "gpt-4-turbo"
    GPT_3_5_TURBO       = "gpt-3.5-turbo"
    GPT_5               = "gpt-5"
    GPT_5_MINI          = "gpt-5-mini"
    GPT_5_NANO          = "gpt-5-nano"
    GPT_5_1             = "gpt-5.1"
    GPT_5_2             = "gpt-5.2"
    GPT_5_4             = "gpt-5.4"
    GPT_5_5             = "gpt-5.5"
    GPT_5_5_PRO         = "gpt-5.5-pro"
    O3                  = "o3"
    O3_PRO              = "o3-pro"
    O3_DEEP_RESEARCH    = "o3-deep-research"
    O4_MINI             = "o4-mini"
    O4_MINI_DR          = "o4-mini-deep-research"
    OSS_20B             = "openai/gpt-oss-20b"
    OSS_120B            = "openai/gpt-oss-120b"

    # ── Meta LLaMA ────────────────────────────────────────────────────────
    LLAMA_4_SCOUT       = "meta/llama-4-scout-instruct"
    LLAMA_4_MAVERICK    = "meta/llama-4-maverick-instruct"
    LLAMA_3_70B         = "meta/meta-llama-3-70b-instruct"
    LLAMA_2_70B         = "meta/llama-2-70b-chat"

    # ── Mistral ───────────────────────────────────────────────────────────
    MISTRAL_LARGE       = "mistral-large-latest"
    MISTRAL_MEDIUM      = "mistral-medium-latest"
    MISTRAL_SMALL       = "mistral-small-latest"
    MISTRAL_NEMO        = "open-mistral-nemo"
    MAGISTRAL_MEDIUM    = "magistral-medium-latest"
    MAGISTRAL_SMALL     = "magistral-small-latest"
    MINISTRAL_14B       = "ministral-14b-latest"

    # ── DeepSeek ──────────────────────────────────────────────────────────
    DEEPSEEK_CHAT       = "deepseek-chat"
    DEEPSEEK_REASONER   = "deepseek-reasoner"

    # ── Perplexity Sonar ──────────────────────────────────────────────────
    SONAR               = "sonar"
    SONAR_PRO           = "sonar-pro"
    SONAR_REASONING_PRO = "sonar-reasoning-pro"
    SONAR_DEEP_RESEARCH = "sonar-deep-research"

    # ── Cohere ────────────────────────────────────────────────────────────
    COMMAND_R           = "command-r-08-2024"

    # ── Default ───────────────────────────────────────────────────────────
    DEFAULT             = "gpt-4.1-nano"


# ──────────────────────────────────────────────────────────────────────────────
# Code generation models  (feature: CODE_GENERATOR)
# ──────────────────────────────────────────────────────────────────────────────

class CodeModel(Enum):
    """Models for ``CODE_GENERATOR``."""

    QWEN_CODER_PLUS     = "qwen3-coder-plus"
    QWEN_CODER_FLASH    = "qwen3-coder-flash"
    CLAUDE_SONNET_4_6   = "claude-sonnet-4-6"
    CLAUDE_OPUS_4_6     = "claude-opus-4-6"
    GPT_5_1_CODEX       = "gpt-5.1-codex"
    GPT_5_1_CODEX_MINI  = "gpt-5.1-codex-mini"
    GPT_4O              = "gpt-4o"
    GROK_CODE           = "grok-code-fast-1"
    DEEPSEEK_REASONER   = "deepseek-reasoner"
    MISTRAL_LARGE       = "mistral-large-latest"

    DEFAULT             = "gpt-4o"


# ──────────────────────────────────────────────────────────────────────────────
# Image generation models  (feature: IMAGE_GENERATOR)
# ──────────────────────────────────────────────────────────────────────────────

class ImageModel(Enum):
    """Models for ``IMAGE_GENERATOR``."""

    # ── Alibaba Qwen Image ────────────────────────────────────────────────
    QWEN_IMAGE_PLUS         = "qwen-image-plus"
    QWEN_IMAGE_MAX          = "qwen-image-max"

    # ── Black Forest Labs Flux ────────────────────────────────────────────
    FLUX_SCHNELL            = "black-forest-labs/flux-schnell"
    FLUX_SCHNELL_LORA       = "black-forest-labs/flux-schnell-lora"
    FLUX_DEV                = "black-forest-labs/flux-dev"
    FLUX_DEV_LORA           = "black-forest-labs/flux-dev-lora"
    FLUX_KREA_DEV           = "black-forest-labs/flux-krea-dev"
    FLUX_1_1_PRO            = "black-forest-labs/flux-1.1-pro"
    FLUX_1_1_PRO_ULTRA      = "black-forest-labs/flux-1.1-pro-ultra"
    FLUX_2_PRO              = "black-forest-labs/flux-2-pro"
    FLUX_2_MAX              = "black-forest-labs/flux-2-max"
    FLUX_2_DEV              = "black-forest-labs/flux-2-dev"
    FLUX_2_FLEX             = "black-forest-labs/flux-2-flex"
    FLUX_2_KLEIN_4B         = "black-forest-labs/flux-2-klein-4b"
    FLUX_2_KLEIN_9B         = "black-forest-labs/flux-2-klein-9b"

    # ── OpenAI GPT Image ──────────────────────────────────────────────────
    GPT_IMAGE_1             = "gpt-image-1"
    GPT_IMAGE_1_MINI        = "gpt-image-1-mini"
    GPT_IMAGE_2             = "gpt-image-2"

    # ── Google Gemini Image ───────────────────────────────────────────────
    GEMINI_2_5_FLASH_IMG    = "gemini-2.5-flash-image"
    GEMINI_3_PRO_IMG        = "gemini-3-pro-image-preview"
    GEMINI_3_1_FLASH_IMG    = "gemini-3.1-flash-image-preview"

    # ── xAI Grok Image ────────────────────────────────────────────────────
    GROK_IMAGINE            = "grok-imagine-image"

    # ── StabilityAI ───────────────────────────────────────────────────────
    STABLE_DIFFUSION_XL     = "clipdrop"
    STABLE_IMAGE_CORE       = "stable-image"
    STABLE_IMAGE_ULTRA      = "stable-image-ultra"
    STABLE_DIFFUSION_XL_1   = "stable-diffusion-xl-1024-v1-0"

    # ── LeonardoAI ────────────────────────────────────────────────────────
    LEONARDO_PHOENIX        = "6b645e3a-d64f-4341-a6d8-7a3690fbf042"
    LEONARDO_LIGHTNING_XL   = "b24e16ff-06e3-43eb-8d33-4416c2d75876"
    LEONARDO_ANIME_XL       = "e71a1c2f-4f80-4800-934f-2c68979d8cc8"
    LEONARDO_KINO_XL        = "aa77f04e-3eec-4034-9c07-d0f619684628"
    LEONARDO_VISION_XL      = "5c232a9e-9061-4777-980a-ddc8e65647c6"
    LEONARDO_ALBEDO_XL      = "2067ae52-33fd-4a82-bb92-c2c55e7d2786"
    LEONARDO_DIFFUSION_XL   = "1e60896f-3c26-4296-8ecc-53e2afecc132"

    # ── Magic Art / Midjourney ────────────────────────────────────────────
    MAGIC_ART_7             = "magic-art_7_0"
    MAGIC_ART_6             = "magic-art_6_1"
    MAGIC_ART_5             = "magic-art"

    # ── Recraft ───────────────────────────────────────────────────────────
    RECRAFT                 = "recraft"

    # ── Dzine ────────────────────────────────────────────────────────────
    DZINE                   = "dzine"

    DEFAULT                 = "black-forest-labs/flux-1.1-pro"


# ──────────────────────────────────────────────────────────────────────────────
# Text-to-speech  (feature: TEXT_TO_SPEECH)
# ──────────────────────────────────────────────────────────────────────────────

class TTSModel(Enum):
    """Models for ``TEXT_TO_SPEECH``."""
    OPENAI_TTS_1        = "tts-1"
    OPENAI_TTS_1_HD     = "tts-1-hd"
    ELEVENLABS          = "elevenlabs-tts"
    GOOGLE              = "google-tts"
    QWEN_TTS_FLASH      = "qwen3-tts-flash"

    DEFAULT             = "tts-1"


class TTSVoice(Enum):
    """Built-in TTS voice names (OpenAI set; ElevenLabs voices are raw strings)."""
    ALLOY   = "alloy"
    ASH     = "ash"
    BALLAD  = "ballad"
    CORAL   = "coral"
    ECHO    = "echo"
    FABLE   = "fable"
    NOVA    = "nova"
    ONYX    = "onyx"
    SAGE    = "sage"
    SHIMMER = "shimmer"
    VERSE   = "verse"


# ──────────────────────────────────────────────────────────────────────────────
# Speech-to-text  (feature: SPEECH_TO_TEXT)
# ──────────────────────────────────────────────────────────────────────────────

class STTModel(Enum):
    """Models for ``SPEECH_TO_TEXT``."""
    WHISPER_1               = "whisper-1"
    GPT_4O_TRANSCRIBE       = "gpt-4o-transcribe"
    GPT_4O_TRANSCRIBE_DIAR  = "gpt-4o-transcribe-diarize"
    ELEVENLABS              = "elevenlabs-speech-to-text"
    QWEN_ASR_FLASH          = "qwen3-asr-flash"
    GOOGLE_LATEST_LONG      = "latest_long"
    GOOGLE_LATEST_SHORT     = "latest_short"
    GOOGLE_MEDICAL_CONV     = "medical_conversation"
    GOOGLE_MEDICAL_DICT     = "medical_dictation"
    GOOGLE_PHONE_CALL       = "phone_call"
    GOOGLE_TELEPHONY        = "telephony"
    GOOGLE_TELEPHONY_SHORT  = "telephony_short"

    DEFAULT                 = "whisper-1"


# ──────────────────────────────────────────────────────────────────────────────
# Video models  (feature: TEXT_TO_VIDEO / IMAGE_TO_VIDEO)
# ──────────────────────────────────────────────────────────────────────────────

class VideoModel(Enum):
    """Models for ``TEXT_TO_VIDEO`` and ``IMAGE_TO_VIDEO``."""
    KLING           = "kling"
    PIKA            = "pika"
    LUMA            = "luma"
    SORA_2          = "sora-2"
    SORA_2_PRO      = "sora-2-pro"
    VEO3            = "veo3-video"
    HAILUO          = "hailuo"
    SKYREELS        = "Qubico/skyreels"
    HUNYUAN         = "Qubico/hunyuan"
    WANX            = "Qubico/wanx"
    TONGYI          = "cjwbw/damo-text-to-video:1e205ea73084bd17a0a3b43396e49ba0d6bc2e754e9283b2df49fad2dcf95755"
    LEONARDO_MOTION = "leonardo-motion"

    DEFAULT         = "kling"


# ──────────────────────────────────────────────────────────────────────────────
# Music models  (feature: MUSIC_GENERATOR)
# ──────────────────────────────────────────────────────────────────────────────

class MusicModel(Enum):
    """Models for ``MUSIC_GENERATOR``."""
    LYRIA_3_PRO     = "lyria-3-pro-preview"
    LYRIA_3_CLIP    = "lyria-3-clip-preview"
    LYRIA_2         = "lyria-002"
    MUSICGEN        = "meta/musicgen:671ac645ce5e552cc63a54a2bbff63fcf798043055d2dac5fc9e36a837eedcfb"
    SUNO            = "suno-ttapi"
    UDIO            = "music-u"

    DEFAULT         = "lyria-002"


# ──────────────────────────────────────────────────────────────────────────────
# Asset types
# ──────────────────────────────────────────────────────────────────────────────

class AssetType(Enum):
    """Asset categories for the team Asset API."""
    IMAGE       = "IMAGE"
    DOCUMENT    = "DOCUMENT"   # txt, pdf, docx, etc.
    AUDIO       = "AUDIO"
    VIDEO       = "VIDEO"


# ──────────────────────────────────────────────────────────────────────────────
# Defaults (plain strings for internal use)
# ──────────────────────────────────────────────────────────────────────────────

DEFAULT_CHAT_MODEL:  str = ChatModel.DEFAULT.value
DEFAULT_CODE_MODEL:  str = CodeModel.DEFAULT.value
DEFAULT_IMAGE_MODEL: str = ImageModel.DEFAULT.value
DEFAULT_TTS_MODEL:   str = TTSModel.DEFAULT.value
DEFAULT_STT_MODEL:   str = STTModel.DEFAULT.value
DEFAULT_VIDEO_MODEL: str = VideoModel.DEFAULT.value
DEFAULT_MUSIC_MODEL: str = MusicModel.DEFAULT.value