# 1minai

An **async Python client** for the [1min.AI](https://1min.ai) API.

```python
import asyncio
import oneminai

async def main():
    async with oneminai.OneMinAIClient("YOUR_API_KEY") as client:
        r = await client.generate_content("What is the capital of Poland?")
        print(r.text)

asyncio.run(main())
```

---

## Features

- **Multi-turn chat** — stateful `ChatSession` keeps context across turns automatically
- **Streaming** — yield tokens in real time via Server-Sent Events
- **Web search** — ground responses with live data, toggleable per-session or per-request
- **Image generation** — 30+ models: DALL-E, Flux 2, Stable Diffusion, Gemini, Grok, Recraft and more
- **Image editing** — upscale, background removal, text removal, variations
- **Image-to-prompt** — reverse-engineer any image into a generation-ready prompt
- **Text to speech** — OpenAI TTS and ElevenLabs voices
- **Speech to text** — Whisper, GPT-4o Transcribe, ElevenLabs, and Google STT
- **Video generation** — Kling, Sora 2, Veo 3, Luma Ray 2, Wan 2.1, and more
- **Music generation** — Google Lyria, Suno, Udio, MusicGen
- **Code generation** — specialised models (GPT Codex, Qwen3 Coder, Grok Code, DeepSeek)
- **Content tools** — grammar, paraphrase, summarise, translate, blog/email/social copy
- **Asset API** — upload local files or URLs (images, PDFs, audio) for use across all endpoints
- **Conversation management** — create, list, and delete server-side threads
- **Credit estimation** — check costs before sending a request
- **Async-first** — built on `aiohttp`, no blocking calls

---

## Installation

Requires Python **3.10+**.

```sh
pip install 1minai
```

---

## Authentication

1. Sign in at [app.1min.ai](https://app.1min.ai).
2. Go to **Settings → API** and copy your API key.
3. Pass it to `OneMinAIClient`.

> Keep your key secret — it grants full access to your 1min.AI account and credits.

Alternatively, authenticate with a Google OAuth token (see [OAuth](#oauth)).

---

## Quick start

```python
import asyncio
import oneminai

async def main():
    async with oneminai.OneMinAIClient("YOUR_API_KEY") as client:
        r = await client.generate_content("Explain async/await in Python.")
        print(r.text)

asyncio.run(main())
```

The client can also be constructed and closed manually:

```python
client = oneminai.OneMinAIClient("YOUR_API_KEY")
r = await client.generate_content("Hello!")
await client.close()
```

---

## Usage

### Single-turn generation

```python
r = await client.generate_content("Translate 'hello' to Japanese.")
print(r.text)           # "こんにちは"
print(r.model)          # "gpt-4.1-nano"
print(str(r))           # same as r.text
```

---

### Streaming

```python
async for chunk in client.generate_content_stream("Write me a haiku."):
    print(chunk.text_delta, end="", flush=True)
print()
```

---

### Multi-turn chat

`start_chat()` returns a `ChatSession` that keeps history automatically:

```python
chat = client.start_chat()

r1 = await chat.send_message("My name is Alice.")
r2 = await chat.send_message("What is my name?")   # → "Your name is Alice."
print(r2.text)
```

Streaming within a session:

```python
async for chunk in chat.send_message_stream("Write me an async Python example."):
    print(chunk.text_delta, end="", flush=True)
print()

# Context is preserved across streaming and non-streaming turns
r3 = await chat.send_message("Now explain what you just wrote.")
```

---

### Web search

```python
# Enable for every turn in the session
chat = client.start_chat(web_search=True, num_of_site=5)
r = await chat.send_message("What AI models were released this week?")

# Or per-request override
chat2 = client.start_chat()
r2 = await chat2.send_message("Latest Python release?", web_search=True)
```

---

### Model selection

```python
from oneminai.constants import ChatModel

r = await client.generate_content("Hello!", model=ChatModel.CLAUDE_SONNET_4_6)

# Session-level default, with per-turn override
chat = client.start_chat(model=ChatModel.GPT_4O)
r2 = await chat.send_message("Quick reply.", model=ChatModel.GPT_4O_MINI)

# Any raw model ID string works too
r3 = await client.generate_content("Hello!", model="gemini-2.5-flash")
```

Available chat models (partial list — any ID from `/models` also works):

| Enum | Model ID |
|---|---|
| `ChatModel.GPT_4O` | `gpt-4o` |
| `ChatModel.GPT_4O_MINI` | `gpt-4o-mini` |
| `ChatModel.GPT_5` | `gpt-5` |
| `ChatModel.GPT_4_1` | `gpt-4.1` |
| `ChatModel.GPT_4_1_NANO` | `gpt-4.1-nano` |
| `ChatModel.O3` | `o3` |
| `ChatModel.O4_MINI` | `o4-mini` |
| `ChatModel.CLAUDE_SONNET_4_6` | `claude-sonnet-4-6` |
| `ChatModel.CLAUDE_OPUS_4_6` | `claude-opus-4-6` |
| `ChatModel.CLAUDE_HAIKU_4_5` | `claude-haiku-4-5-20251001` |
| `ChatModel.GEMINI_2_5_PRO` | `gemini-2.5-pro` |
| `ChatModel.GEMINI_2_5_FLASH` | `gemini-2.5-flash` |
| `ChatModel.GROK_4` | `grok-4-0709` |
| `ChatModel.DEEPSEEK_CHAT` | `deepseek-chat` |
| `ChatModel.DEEPSEEK_REASONER` | `deepseek-reasoner` |
| `ChatModel.LLAMA_4_SCOUT` | `meta/llama-4-scout-instruct` |
| `ChatModel.SONAR_PRO` | `sonar-pro` |
| `ChatModel.MISTRAL_LARGE` | `mistral-large-latest` |

---

### Persistent server-side conversations

```python
# Create a named thread
conv = await client.create_conversation("Research session")

# Resume it by passing the conversation_id
reply = await client.chat(
    "What is quantum entanglement?",
    conversation_id=conv.conversation_id,
)

# Fetch history
messages = await client.get_conversation_messages(conv.conversation_id)
for msg in messages:
    print(f"[{msg.role}]: {msg.content}")

# List all threads
all_convs = await client.list_conversations()

# Clean up
await client.delete_conversation(conv.conversation_id)
```

---

### Image generation

```python
from oneminai.constants import ImageModel

result = await client.generate_image(
    "A serene mountain lake at golden hour, photorealistic",
    model=ImageModel.FLUX_1_1_PRO,
    num_images=2,
    aspect_ratio="16:9",
)

print(result.image.url)          # first image URL
for img in result.images:
    await img.save("./outputs")  # download all
```

Available image models include Flux 1.1 Pro / 2 Pro, DALL-E / GPT-Image, Gemini Image, Grok Imagine, Stable Diffusion XL, Leonardo, Recraft, and more.

---

### Image editing

```python
# Upload the source image
asset = await client.upload_asset("photo.jpg")

# Upscale
upscaled = await client.upscale_image(asset.asset_key, scale=2)

# Remove background
no_bg = await client.remove_background(asset.asset_key)

# Generate variations
variants = await client.image_variations(
    asset.asset_key,
    prompt="same composition, cyberpunk neon lighting",
    num_images=4,
)

# Convert image → generation prompt
prompt_text = await client.image_to_prompt(asset.asset_key)
print(prompt_text.text)
```

---

### Text to speech

```python
from oneminai.constants import TTSModel, TTSVoice

audio = await client.text_to_speech(
    "Hello, this is a test.",
    model=TTSModel.OPENAI_TTS_1_HD,
    voice=TTSVoice.NOVA,
)
await audio.save("./outputs", verbose=True)

# ElevenLabs
el = await client.text_to_speech(
    "Sounds natural and expressive.",
    model=TTSModel.ELEVENLABS,
    voice="Rachel",          # ElevenLabs voices are raw strings
)

# Sound effects
sfx = await client.text_to_sound(
    "Heavy rain on a tin roof, distant thunder",
    duration=10.0,
)
```

---

### Speech to text

```python
from oneminai.constants import STTModel

asset = await client.upload_asset("recording.mp3", asset_type=oneminai.AssetType.AUDIO)

result = await client.transcribe(asset.asset_key, model=STTModel.WHISPER_1)
print(result.text)

# Or from a URL
result2 = await client.transcribe_url(
    "https://example.com/audio.mp3",
    model=STTModel.GPT_4O_TRANSCRIBE,
)
```

---

### Video generation

```python
from oneminai.constants import VideoModel

# Text to video
video = await client.generate_video(
    "Drone shot rising over snow-capped mountains at dawn",
    model=VideoModel.KLING,
)
await video.save("./outputs")

# Image to video
asset = await client.upload_asset("frame.jpg")
vid2 = await client.image_to_video(
    asset.asset_key,
    prompt="Camera slowly pans left, leaves gently swaying",
    model=VideoModel.LUMA,
)
```

---

### Music generation

```python
from oneminai.constants import MusicModel

track = await client.generate_music(
    "Upbeat lo-fi hip hop, 90 BPM, warm piano chords",
    model=MusicModel.LYRIA_2,
)
await track.save("./outputs")

# Instrumental
score = await client.generate_music(
    "Epic cinematic orchestral, building tension",
    instrumental=True,
    duration=30.0,
)
```

---

### Code generation

```python
from oneminai.constants import CodeModel

result = await client.generate_code(
    "Async HTTP client with exponential-backoff retry, using aiohttp.",
    model=CodeModel.CLAUDE_SONNET_4_6,
)
print(result.text)
```

---

### Content tools

```python
# Grammar
corrected  = await client.check_grammar("its a beautifull day!")
paraphrased = await client.paraphrase("Text to rewrite.", tone="casual")
rewritten   = await client.rewrite("Formal text.", tone="friendly")
summary     = await client.summarize(long_article)
expanded    = await client.expand_content("Short stub.")
shortened   = await client.shorten_content(verbose_text)

# Translation
translated = await client.translate("Good morning", "Polish")

# SEO
keywords = await client.research_keywords("async Python AI development")

# Content generation
article  = await client.generate_blog_article("AI trends in 2025", tone="professional")
email    = await client.generate_email("Partnership proposal to Acme Corp")
reply    = await client.generate_email_reply(original_email, "Accept politely")
tweet    = await client.generate_social_post("Launch day!", platform="x")
linkedin = await client.generate_social_post("We shipped v1!", platform="linkedin")
```

---

### Asset upload

```python
# Local file
asset = await client.upload_asset("report.pdf", asset_type=oneminai.AssetType.DOCUMENT)

# Raw bytes
asset2 = await client.upload_asset(
    data=some_bytes, filename="image.png", mime_type="image/png"
)

# Public URL
asset3 = await client.upload_asset_from_url("https://example.com/photo.jpg")

# Use in chat
reply = await client.generate_content(
    "Summarise this document.",
    files=[asset.file_id],
)

# Use in image API
result = await client.upscale_image(asset3.asset_key)
```

---

### Credits & account

```python
user = await client.get_current_user()
print(user.email, user.plan, user.credit)

balance = await client.get_team_credits()

# Estimate before sending
estimate = await client.estimate_chat_cost(
    "Write a long essay.",
    models=["gpt-4o", "claude-sonnet-4-6"],
    web_search=True,
)
print(estimate.total_estimated_credit, "credits")
```

---

### OAuth

```python
client = oneminai.OneMinAIClient()
user = await client.oauth_login("ya29.a0AQvPy…")   # Google access token
print(f"Authenticated as {user.email}")

r = await client.generate_content("Hello!")
await client.close()
```

---

### Error handling

```python
from oneminai import (
    AuthenticationError,
    RateLimitError,
    ValidationError,
    APIError,
)

try:
    r = await client.generate_content("Hello!")
except AuthenticationError:
    print("Invalid API key.")
except RateLimitError as e:
    print(f"Rate limited — retry in {e.retry_after_s}s")
except ValidationError as e:
    print(f"Validation error: {e.details}")
except APIError as e:
    print(f"API error {e.status_code}: {e}")
```

---

### Logging

```python
import oneminai
oneminai.set_log_level("DEBUG")   # "DEBUG" | "INFO" | "WARNING" | "ERROR"
```

---

## API reference

### `OneMinAIClient`

| Method | Description |
|---|---|
| `generate_content(prompt, ...)` | Single-turn text generation |
| `generate_content_stream(prompt, ...)` | Single-turn streaming |
| `chat(prompt, *, stream, ...)` | Low-level chat (stream or not) |
| `start_chat(model, ...)` | Create a `ChatSession` |
| `generate_image(prompt, ...)` | Image generation |
| `upscale_image(asset_key, ...)` | Upscale an image |
| `remove_background(asset_key)` | Remove image background |
| `replace_background(asset_key, ...)` | Replace background with prompt |
| `image_variations(asset_key, ...)` | Generate image variations |
| `image_to_prompt(asset_key)` | Reverse-engineer an image to a prompt |
| `text_to_speech(text, ...)` | Text to speech |
| `text_to_sound(prompt, ...)` | Sound effect generation |
| `transcribe(asset_key, ...)` | Speech to text (uploaded asset) |
| `transcribe_url(url, ...)` | Speech to text (from URL) |
| `generate_video(prompt, ...)` | Text-to-video generation |
| `image_to_video(asset_key, ...)` | Image-to-video generation |
| `generate_music(prompt, ...)` | Music generation |
| `generate_code(prompt, ...)` | Code generation |
| `check_grammar(text)` | Grammar correction |
| `paraphrase(text, ...)` | Paraphrase with tone control |
| `rewrite(text, ...)` | Rewrite with tone control |
| `summarize(text, ...)` | Summarisation |
| `translate(text, target_language)` | Translation |
| `generate_blog_article(prompt, ...)` | Blog article generation |
| `generate_email(prompt, ...)` | Email generation |
| `generate_email_reply(original, instructions, ...)` | Email reply generation |
| `generate_social_post(prompt, platform, ...)` | Social media post |
| `upload_asset(file_path, ...)` | Upload a local file |
| `upload_asset_from_url(url, ...)` | Register a URL as an asset |
| `create_conversation(title)` | Create a server-side thread |
| `list_conversations()` | List all threads |
| `get_conversation_messages(conv_id)` | Fetch conversation history |
| `delete_conversation(conv_id)` | Delete a thread |
| `get_current_user()` | Fetch user and team info |
| `get_team_credits()` | Current credit balance |
| `estimate_chat_cost(prompt, models, ...)` | Pre-flight credit estimate |
| `list_models(feature)` | Model catalog |
| `oauth_login(token)` | Authenticate with Google OAuth |

### `ChatSession`

| Method / Property | Description |
|---|---|
| `send_message(prompt, ...)` | Send a message and await the full reply |
| `send_message_stream(prompt, ...)` | Send and stream the reply |
| `delete()` | Delete the server-side conversation |
| `.conversation_id` | Server UUID of this thread |
| `.model` | Model used by this session |
| `.last_output` | Most recent `ChatOutput` |

---

## Development

```sh
git clone https://github.com/your-org/1minai
cd 1minai
pip install -e ".[dev]"

# Run examples
export ONEMINAI_API_KEY="your-key"
python examples.py
```

---

## References

- [1min.AI](https://1min.ai)
- [1min.AI API docs](https://docs.1min.ai/docs/api/intro)
