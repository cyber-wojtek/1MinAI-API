"""
examples.py — Runnable examples for the 1minai Python client.

Each example is a self-contained async function.  Run the whole file:

    python examples.py

Or run a single example by calling it directly at the bottom.

Set your API key via the environment variable:

    export ONEMINAI_API_KEY="your-key-here"
"""

from __future__ import annotations

import asyncio
import os

import oneminai_webapi
from oneminai_webapi import OneMinAIClient
from oneminai_webapi.constants import (
    ChatModel,
    CodeModel,
    ImageModel,
    MusicModel,
    STTModel,
    TTSModel,
    TTSVoice,
    VideoModel,
)

API_KEY = os.environ.get("ONEMINAI_API_KEY", "...your-api-key-here...")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  BASIC GENERATION
# ══════════════════════════════════════════════════════════════════════════════

async def example_basic_generation() -> None:
    """Single-turn text generation — the simplest possible call."""
    async with OneMinAIClient(API_KEY) as client:
        response = await client.generate_content("What is the capital of Poland?")
        print("Response:", response.text)
        print("Model used:", response.model)
        # str(response) is equivalent to response.text
        assert str(response) == response.text


# ══════════════════════════════════════════════════════════════════════════════
# 2.  STREAMING
# ══════════════════════════════════════════════════════════════════════════════

async def example_streaming() -> None:
    """Stream tokens as they arrive — ideal for long outputs."""
    async with OneMinAIClient(API_KEY) as client:
        print("Streaming: ", end="", flush=True)
        async for chunk in client.generate_content_stream(
            "Write a haiku about asynchronous Python."
        ):
            print(chunk.text_delta, end="", flush=True)
        print()  # newline after stream ends


# ══════════════════════════════════════════════════════════════════════════════
# 3.  MODEL SELECTION
# ══════════════════════════════════════════════════════════════════════════════

async def example_model_selection() -> None:
    """Choose any model by enum or raw string."""
    async with OneMinAIClient(API_KEY) as client:
        # Via enum
        r1 = await client.generate_content(
            "Summarise quantum entanglement in one sentence.",
            model=ChatModel.CLAUDE_SONNET_4_6,
        )
        print("Claude:", r1.text)

        # Via raw string — any model ID from the catalog works
        r2 = await client.generate_content(
            "Same question.", model="gemini-2.5-flash"
        )
        print("Gemini:", r2.text)

        # List all available models
        models = await client.list_models(feature="UNIFY_CHAT_WITH_AI")
        print(f"Available chat models: {len(models)}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  MULTI-TURN CHAT SESSION
# ══════════════════════════════════════════════════════════════════════════════

async def example_chat_session() -> None:
    """Stateful multi-turn conversation — the client tracks context."""
    async with OneMinAIClient(API_KEY) as client:
        chat = client.start_chat(model=ChatModel.GPT_4O_MINI)

        r1 = await chat.send_message("My name is Alice and I love Python.")
        print("Turn 1:", r1.text)

        r2 = await chat.send_message("What's my name and favourite language?")
        print("Turn 2:", r2.text)  # should reference "Alice" and "Python"

        # Mix streaming and non-streaming within one session
        print("Streaming turn: ", end="", flush=True)
        async for chunk in chat.send_message_stream("Give me a one-liner about it."):
            print(chunk.text_delta, end="", flush=True)
        print()

        # Session conversation ID is persisted automatically
        print("Conversation ID:", chat.conversation_id)


# ══════════════════════════════════════════════════════════════════════════════
# 5.  WEB SEARCH GROUNDING
# ══════════════════════════════════════════════════════════════════════════════

async def example_web_search() -> None:
    """Ground responses with live web data."""
    async with OneMinAIClient(API_KEY) as client:
        # Enable for every turn in the session
        chat = client.start_chat(web_search=True, num_of_site=5)
        r = await chat.send_message("What AI models were released this week?")
        print("Web-grounded:", r.text[:300])

        # Or enable per-request only
        chat2 = client.start_chat()
        r2 = await chat2.send_message(
            "Latest Python release?", web_search=True
        )
        print("Per-request search:", r2.text[:300])


# ══════════════════════════════════════════════════════════════════════════════
# 6.  PERSISTENT SERVER-SIDE CONVERSATIONS
# ══════════════════════════════════════════════════════════════════════════════

async def example_conversations() -> None:
    """Create, list, and delete server-side conversation threads."""
    async with OneMinAIClient(API_KEY) as client:
        # Create a named thread
        conv = await client.create_conversation("My research session")
        print("Created:", conv.conversation_id, conv.title)

        # Chat within it
        r = await client.chat(
            "Tell me about black holes.",
            conversation_id=conv.conversation_id,
        )
        print("Reply:", r.text[:200])

        # List all conversations
        all_convs = await client.list_conversations()
        print(f"Total conversations: {len(all_convs)}")

        # Fetch history
        messages = await client.get_conversation_messages(conv.conversation_id)
        for msg in messages:
            print(f"  [{msg.role}]: {msg.content[:80]}")

        # Tidy up
        await client.delete_conversation(conv.conversation_id)
        print("Deleted conversation.")


# ══════════════════════════════════════════════════════════════════════════════
# 7.  IMAGE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

async def example_image_generation() -> None:
    """Generate images with various models."""
    async with OneMinAIClient(API_KEY) as client:
        # Basic generation
        result = await client.generate_image(
            "A serene mountain lake at golden hour, photorealistic",
            model=ImageModel.FLUX_1_1_PRO,
        )
        print(f"Generated {len(result.images)} image(s)")
        print("First image URL:", result.image.url)

        # Save locally
        path = await result.image.save("./outputs", verbose=True)
        print("Saved to:", path)

        # Multiple images
        result2 = await client.generate_image(
            "Abstract geometric art, vibrant colors",
            model=ImageModel.FLUX_2_PRO,
            num_images=4,
            aspect_ratio="16:9",
        )
        for i, img in enumerate(result2.images):
            print(f"  Image {i+1}: {img.url}")


# ══════════════════════════════════════════════════════════════════════════════
# 8.  IMAGE EDITING
# ══════════════════════════════════════════════════════════════════════════════

async def example_image_editing() -> None:
    """Upload an image asset and apply edits."""
    async with OneMinAIClient(API_KEY) as client:
        # Upload the source image first
        asset = await client.upload_asset(
            "my_photo.jpg"  # path to a local file
        )
        print("Uploaded asset key:", asset.asset_key)

        # Upscale
        upscaled = await client.upscale_image(asset.asset_key, scale=2)
        print("Upscaled:", upscaled.image.url)

        # Remove background
        no_bg = await client.remove_background(asset.asset_key)
        print("No-BG:", no_bg.image.url)

        # Image variations
        variants = await client.image_variations(
            asset.asset_key,
            prompt="same composition, cyberpunk neon lighting",
            num_images=2,
        )
        for v in variants.images:
            print("Variant:", v.url)

        # Convert image to a generation prompt
        prompt_result = await client.image_to_prompt(asset.asset_key)
        print("Reverse prompt:", prompt_result.text)


# ══════════════════════════════════════════════════════════════════════════════
# 9.  TEXT TO SPEECH
# ══════════════════════════════════════════════════════════════════════════════

async def example_text_to_speech() -> None:
    """Convert text to speech using OpenAI TTS or ElevenLabs."""
    async with OneMinAIClient(API_KEY) as client:
        # OpenAI TTS
        audio = await client.text_to_speech(
            "Hello, this is a test of the 1min.AI text-to-speech API.",
            model=TTSModel.OPENAI_TTS_1_HD,
            voice=TTSVoice.NOVA,
        )
        print("Audio URL:", audio.audio_url)
        path = await audio.save("./outputs", verbose=True)
        print("Saved to:", path)

        # ElevenLabs with a named voice
        el_audio = await client.text_to_speech(
            "ElevenLabs sounds natural and expressive.",
            model=TTSModel.ELEVENLABS,
            voice="Rachel",
        )
        print("ElevenLabs audio:", el_audio.audio_url)

        # Sound effect generation
        sfx = await client.text_to_sound(
            "Heavy rain on a tin roof, thunder rumbling in the distance",
            duration=10.0,
        )
        print("Sound effect:", sfx.audio_url)


# ══════════════════════════════════════════════════════════════════════════════
# 10.  SPEECH TO TEXT
# ══════════════════════════════════════════════════════════════════════════════

async def example_speech_to_text() -> None:
    """Transcribe audio files using Whisper or other STT models."""
    async with OneMinAIClient(API_KEY) as client:
        # Upload audio asset first
        asset = await client.upload_asset(
            "recording.mp3",
            asset_type=oneminai_webapi.AssetType.AUDIO,
        )
        # Transcribe
        result = await client.transcribe(
            asset.asset_key,
            model=STTModel.WHISPER_1,
            language="en",
        )
        print("Transcript:", result.text)

        # Or transcribe directly from a URL
        result2 = await client.transcribe_url(
            "https://example.com/audio.mp3",
            model=STTModel.GPT_4O_TRANSCRIBE,
        )
        print("URL transcript:", result2.text)


# ══════════════════════════════════════════════════════════════════════════════
# 11.  VIDEO GENERATION
# ══════════════════════════════════════════════════════════════════════════════

async def example_video_generation() -> None:
    """Generate video from text or image."""
    async with OneMinAIClient(API_KEY) as client:
        # Text-to-video
        video = await client.generate_video(
            "A drone shot slowly rising over a snow-capped mountain range at dawn",
            model=VideoModel.KLING,
        )
        print("Video URL:", video.video_url)
        path = await video.save("./outputs", verbose=True)
        print("Saved to:", path)

        # Image-to-video: upload a reference image first
        asset = await client.upload_asset("reference.jpg")
        vid2 = await client.image_to_video(
            asset.asset_key,
            prompt="The camera slowly pans left, leaves gently swaying",
            model=VideoModel.LUMA,
        )
        print("Image-to-video:", vid2.video_url)


# ══════════════════════════════════════════════════════════════════════════════
# 12.  MUSIC GENERATION
# ══════════════════════════════════════════════════════════════════════════════

async def example_music_generation() -> None:
    """Generate original music from a text description."""
    async with OneMinAIClient(API_KEY) as client:
        track = await client.generate_music(
            "Upbeat lo-fi hip hop, calm and focused, 90 BPM, warm piano chords",
            model=MusicModel.LYRIA_2,
        )
        print("Music URL:", track.audio_url)
        path = await track.save("./outputs", verbose=True)
        print("Saved to:", path)

        # Instrumental (no vocals)
        instrumental = await client.generate_music(
            "Epic orchestral score, cinematic, building tension",
            instrumental=True,
            duration=30.0,
        )
        print("Instrumental:", instrumental.audio_url)


# ══════════════════════════════════════════════════════════════════════════════
# 13.  CODE GENERATION
# ══════════════════════════════════════════════════════════════════════════════

async def example_code_generation() -> None:
    """Use specialised coding models for code generation."""
    async with OneMinAIClient(API_KEY) as client:
        result = await client.generate_code(
            "Write a Python async HTTP client that retries on 5xx errors "
            "using aiohttp and exponential backoff.",
            model=CodeModel.CLAUDE_SONNET_4_6,
        )
        print(result.text)

        # With web search to reference latest library docs
        result2 = await client.generate_code(
            "FastAPI endpoint that validates a JWT token using PyJWT 2.x.",
            web_search=True,
        )
        print(result2.text)


# ══════════════════════════════════════════════════════════════════════════════
# 14.  CONTENT TOOLS
# ══════════════════════════════════════════════════════════════════════════════

async def example_content_tools() -> None:
    """Grammar, translation, summarisation, and content generation."""
    async with OneMinAIClient(API_KEY) as client:
        sample = (
            "the quick brown fox jumps over the lazy dog. "
            "its a beautiful day in the neighbourhood!"
        )

        # Grammar correction
        corrected = await client.check_grammar(sample)
        print("Corrected:", corrected.text)

        # Paraphrase
        paraphrased = await client.paraphrase(sample, tone="casual")
        print("Paraphrased:", paraphrased.text)

        # Summarise
        long_text = "Artificial intelligence (AI) is intelligence demonstrated by " * 20
        summary = await client.summarize(long_text)
        print("Summary:", summary.text)

        # Translate
        translated = await client.translate("Good morning, how are you?", "Polish")
        print("Polish:", translated.text)

        # Generate a blog article
        article = await client.generate_blog_article(
            "The future of async Python for AI applications",
            tone="professional",
            language="English",
        )
        print("Article preview:", article.text[:300])

        # Social post
        tweet = await client.generate_social_post(
            "Excited to launch our new open-source Python AI client!",
            platform="x",
            tone="enthusiastic",
        )
        print("Tweet:", tweet.text)

        # LinkedIn post
        linkedin = await client.generate_social_post(
            "We just launched 1minai — an async Python client for 1min.AI.",
            platform="linkedin",
            tone="professional",
        )
        print("LinkedIn:", linkedin.text)


# ══════════════════════════════════════════════════════════════════════════════
# 15.  ASSET MANAGEMENT
# ══════════════════════════════════════════════════════════════════════════════

async def example_assets() -> None:
    """Upload files or URLs for use across all API endpoints."""
    async with OneMinAIClient(API_KEY) as client:
        # Upload a local PDF
        doc_asset = await client.upload_asset(
            "report.pdf",
            asset_type=oneminai_webapi.AssetType.DOCUMENT,
        )
        print("PDF asset:", doc_asset.file_id)

        # Use the uploaded document in a chat message
        reply = await client.generate_content(
            "Summarise the key findings from this document.",
            files=[doc_asset.file_id],
        )
        print("Document summary:", reply.text[:300])

        # Register a URL as an image asset
        img_asset = await client.upload_asset_from_url(
            "https://upload.wikimedia.org/wikipedia/commons/thumb/2/2f/"
            "Culinary_fruits_front_view.jpg/800px-Culinary_fruits_front_view.jpg"
        )
        print("URL image asset:", img_asset.asset_key)

        # Describe the image via chat
        vision_reply = await client.generate_content(
            "Describe what you see in this image.",
            images=[img_asset.asset_key],
        )
        print("Vision reply:", vision_reply.text)


# ══════════════════════════════════════════════════════════════════════════════
# 16.  ACCOUNT & CREDITS
# ══════════════════════════════════════════════════════════════════════════════

async def example_account() -> None:
    """Fetch user info and check credit balance."""
    async with OneMinAIClient(API_KEY) as client:
        user = await client.get_current_user()
        print(f"Logged in as: {user.email}")
        print(f"Plan: {user.plan}  |  Credits: {user.credit}")
        print(f"Team: {user.team_name} ({user.team_id})")

        balance = await client.get_team_credits()
        print(f"Current credit balance: {balance}")

        # Estimate cost before sending
        estimate = await client.estimate_chat_cost(
            "Write a long essay about quantum computing.",
            models=["gpt-4o", "claude-sonnet-4-6"],
            web_search=True,
        )
        print("Estimated cost:", estimate.total_estimated_credit, "credits")
        print("Per-model breakdown:", estimate.models)


# ══════════════════════════════════════════════════════════════════════════════
# 17.  OAUTH AUTHENTICATION  (alternative to API key)
# ══════════════════════════════════════════════════════════════════════════════

async def example_oauth() -> None:
    """Authenticate with a Google OAuth token instead of an API key."""
    google_oauth_token = "ya29.a0AQvPy..."  # your Google access token

    client = OneMinAIClient()  # no API key needed yet
    user = await client.oauth_login(google_oauth_token)
    print(f"Authenticated as: {user.email}")

    # All subsequent calls work normally
    r = await client.generate_content("Hello!")
    print(r.text)

    await client.close()


# ══════════════════════════════════════════════════════════════════════════════
# 18.  ERROR HANDLING
# ══════════════════════════════════════════════════════════════════════════════

async def example_error_handling() -> None:
    """Demonstrate exception types and how to handle them."""
    from oneminai import (
        APIError,
        AuthenticationError,
        RateLimitError,
        ValidationError,
    )

    try:
        async with OneMinAIClient("INVALID_KEY") as client:
            await client.generate_content("Hello!")
    except AuthenticationError as e:
        print(f"Auth failed: {e}")

    async with OneMinAIClient(API_KEY) as client:
        try:
            await client.generate_content("Hello!", model="non-existent-model-xyz")
        except (ValidationError, APIError) as e:
            print(f"API error ({type(e).__name__}): {e}")
        except RateLimitError as e:
            print(f"Rate limited.  Retry after {e.retry_after_s}s")


# ══════════════════════════════════════════════════════════════════════════════
# 19.  LOGGING
# ══════════════════════════════════════════════════════════════════════════════

async def example_logging() -> None:
    """Enable debug-level logging to trace HTTP calls."""
    oneminai_webapi.set_log_level("DEBUG")

    async with OneMinAIClient(API_KEY) as client:
        r = await client.generate_content("Say hi.")
        print(r.text)

    oneminai_webapi.set_log_level("WARNING")  # quiet again


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════

async def main() -> None:
    """Run all examples sequentially.  Comment out any you want to skip."""
    print("\n── 1. Basic generation ──────────────────────────────────")
    await example_basic_generation()

    print("\n── 2. Streaming ─────────────────────────────────────────")
    await example_streaming()

    print("\n── 3. Model selection ───────────────────────────────────")
    await example_model_selection()

    print("\n── 4. Multi-turn chat ───────────────────────────────────")
    await example_chat_session()

    print("\n── 5. Web search ────────────────────────────────────────")
    await example_web_search()

    print("\n── 16. Account & credits ────────────────────────────────")
    await example_account()

    # The examples below require local files (images, audio, etc.)
    # Uncomment as needed:
    # await example_image_generation()
    # await example_image_editing()
    # await example_text_to_speech()
    # await example_speech_to_text()
    # await example_video_generation()
    # await example_music_generation()
    # await example_code_generation()
    # await example_content_tools()
    # await example_assets()
    # await example_error_handling()
    # await example_logging()


if __name__ == "__main__":
    asyncio.run(main())
