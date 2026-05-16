"""
OneMinAIClient — async client for the 1min.AI API.
"""
from __future__ import annotations

import json
import logging
import mimetypes
import time
import random
import uuid as _uuid
from pathlib import Path
from typing import AsyncIterator, Literal, overload

import aiohttp

from .constants import (
    APP_VERSION,
    BASE_URL,
    DEFAULT_CHAT_MODEL,
    DEFAULT_CODE_MODEL,
    DEFAULT_IMAGE_MODEL,
    DEFAULT_MUSIC_MODEL,
    DEFAULT_STT_MODEL,
    DEFAULT_TTS_MODEL,
    DEFAULT_VIDEO_MODEL,
    AssetType,
    ChatModel,
    CodeModel,
    ImageModel,
    MusicModel,
    STTModel,
    TTSModel,
    TTSVoice,
    VideoModel,
)
from .exceptions import (
    APIError,
    AssetUploadError,
    AuthenticationError,
    OAuthError,
    RateLimitError,
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

logger = logging.getLogger("1minai")

# Fixed endpoints (not team-scoped)
_OAUTH_URL  = f"{BASE_URL}/auth/oauth"
_MODELS_URL = f"{BASE_URL}/models"
_USERS_URL  = f"{BASE_URL}/users"

# Team-scoped URL templates (formatted with team_id at call time)
_TEAM_BASE          = f"{BASE_URL}/teams/{{team_id}}"
_CHAT_V2_URL        = f"{_TEAM_BASE}/features/v2/unified-chat"
_ASSET_URL          = f"{_TEAM_BASE}/assets"
_CONV_URL           = f"{_TEAM_BASE}/features/conversations"
_CONV_DETAIL_URL    = f"{_TEAM_BASE}/features/conversations/{{conv_id}}"
_MSG_URL            = f"{_TEAM_BASE}/features/unified-chat/conversations/messages/after-id"
_FEATURE_URL        = f"{_TEAM_BASE}/features"          # POST for non-chat features
_CREDITS_URL        = f"{_TEAM_BASE}/credits"
_ESTIMATE_URL       = f"{_TEAM_BASE}/credits/estimate-chat"
_SETTINGS_URL       = f"{_TEAM_BASE}/features/settings/{{feature}}"
_MEMBERS_URL        = f"{_TEAM_BASE}/members"
_TAGS_URL           = f"{_TEAM_BASE}/tags"
_NOTIF_UNREAD_URL   = f"{BASE_URL}/notifications/unread"
_NOTIF_DETAIL_URL   = f"{BASE_URL}/notifications/{{notif_id}}"
_USER_SETTINGS_URL  = f"{BASE_URL}/users/settings"
_NOTEBOOK_URL       = f"{BASE_URL}/users/notebook"
_EXPLORE_URL        = f"{BASE_URL}/posts/explore"


# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

def _resolve(
    model: str | ChatModel | CodeModel | ImageModel | TTSModel
           | STTModel | VideoModel | MusicModel | None,
    default: str,
) -> str:
    """Normalise a model argument to a plain string."""
    if model is None:
        return default
    if isinstance(model, (
        ChatModel, CodeModel, ImageModel, TTSModel, STTModel, VideoModel, MusicModel
    )):
        return model.value
    return str(model)


def _message_group() -> str:
    """
    Generate the ``metadata.messageGroup`` token the web app sends with every
    chat request.  Format observed in HAR: ``"{epoch_ms}_{2-digit-random}"``.
    """
    return f"{int(time.time() * 1000)}_{random.randint(10, 99)}"


def _extract_result_urls(ai_record: dict) -> list[str]:
    result = ai_record.get("aiRecordDetail", {}).get("resultObject", [])
    if isinstance(result, list):
        return [str(r) for r in result if r]
    if isinstance(result, str) and result:
        return [result]
    return []


def _text_from_record(record: dict) -> str:
    results = record.get("aiRecordDetail", {}).get("resultObject", [])
    return results[0] if isinstance(results, list) and results else ""


# ──────────────────────────────────────────────────────────────────────────────
# OneMinAIClient
# ──────────────────────────────────────────────────────────────────────────────

class OneMinAIClient:
    """
    Async client for the 1min.AI REST API.

    Parameters
    ----------
    api_key:
        Your 1min.AI API key **or** a JWT token obtained via
        :meth:`oauth_login`.  Omit only when you call :meth:`oauth_login`
        immediately after construction.
    proxy:
        Optional HTTP/S proxy URL, e.g. ``"http://user:pass@host:port"``.
    timeout:
        Default request timeout in seconds (default: 60).

    Examples
    --------
    Key-based::

        async with OneMinAIClient("YOUR_API_KEY") as client:
            r = await client.generate_content("Hello!")
            print(r.text)

    OAuth::

        client = OneMinAIClient()
        user = await client.oauth_login("ya29.a0AQvPy...")
        r = await client.generate_content("Hello!")
    """

    def __init__(
        self,
        api_key: str = "",
        proxy: str | None = None,
        timeout: int = 60,
    ) -> None:
        self._api_key  = api_key
        self._proxy    = proxy
        self._timeout  = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

        # Resolved lazily on the first request that needs it.
        self._team_id: str | None = None
        self._user: UserRecord | None = None

    # ── lifecycle ──────────────────────────────────────────────────────────

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            if not self._api_key:
                raise AuthenticationError(
                    "No API key set.  Call oauth_login() first or pass "
                    "api_key= to OneMinAIClient()."
                )
            self._session = aiohttp.ClientSession(
                headers={
                    "API-KEY":       self._api_key,
                    "Content-Type":  "application/json",
                    "Accept":        "application/json, text/plain, */*",
                    "X-Auth-Token":  "Bearer",
                    "X-App-Version": APP_VERSION,
                },
                connector=aiohttp.TCPConnector(ssl=True),
            )
        return self._session

    async def close(self) -> None:
        """Explicitly close the underlying HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
            logger.info("OneMinAIClient session closed.")

    async def __aenter__(self) -> "OneMinAIClient":
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    # ── team-ID resolution ─────────────────────────────────────────────────

    async def _get_team_id(self) -> str:
        """
        Return the primary team UUID, fetching and caching it on first call.

        The team ID is required for all team-scoped endpoints.  It is derived
        from ``GET /users`` → ``user.teams[0].team.uuid``.
        """
        if self._team_id:
            return self._team_id
        user = await self.get_current_user()
        return user.team_id

    def _team_url(self, template: str, **extra: str) -> str:
        """Format a team-scoped URL template, injecting the cached team ID."""
        if not self._team_id:
            raise RuntimeError(
                "_team_url called before team_id was resolved.  "
                "Await get_current_user() or any API method first."
            )
        return template.format(team_id=self._team_id, **extra)

    # ── error handling ─────────────────────────────────────────────────────

    @staticmethod
    async def _raise_for_status(resp: aiohttp.ClientResponse) -> None:
        if resp.status == 401:
            raise AuthenticationError("Invalid or missing API key / token.")
        if resp.status == 422:
            body = await resp.json(content_type=None)
            err  = body.get("error", {})
            raise ValidationError(
                err.get("message", "Validation failed."),
                details=err.get("details", []),
            )
        if resp.status == 429:
            ra = resp.headers.get("Retry-After")
            raise RateLimitError(
                "1min.AI rate limit exceeded.",
                retry_after_s=int(ra) if ra and ra.isdigit() else None,
            )
        if resp.status >= 400:
            body = await resp.text()
            raise APIError(f"HTTP {resp.status}: {body[:400]}", status_code=resp.status)

    # ── low-level helpers ──────────────────────────────────────────────────

    async def _post(self, url: str, payload: dict, **kwargs: object) -> dict:
        session = await self._get_session()
        async with session.post(
            url,
            data=json.dumps(payload).encode(),
            timeout=self._timeout,
            proxy=self._proxy,
            **kwargs,
        ) as resp:
            await self._raise_for_status(resp)
            return await resp.json(content_type=None)

    async def _get_json(self, url: str, **kwargs: object) -> dict | list:
        session = await self._get_session()
        async with session.get(
            url, timeout=self._timeout, proxy=self._proxy, **kwargs
        ) as resp:
            await self._raise_for_status(resp)
            return await resp.json(content_type=None)

    async def _put(self, url: str, payload: dict) -> dict:
        session = await self._get_session()
        async with session.put(
            url,
            data=json.dumps(payload).encode(),
            timeout=self._timeout,
            proxy=self._proxy,
        ) as resp:
            await self._raise_for_status(resp)
            return await resp.json(content_type=None)

    async def _delete(self, url: str) -> None:
        session = await self._get_session()
        async with session.delete(
            url, timeout=self._timeout, proxy=self._proxy
        ) as resp:
            await self._raise_for_status(resp)

    # ── streaming helper ───────────────────────────────────────────────────

    async def _post_stream(self, url: str, payload: dict) -> AsyncIterator[dict]:
        """
        POST *payload* and yield parsed SSE event dicts
        ``{"event": str, "data": dict}``.
        """
        session = await self._get_session()
        async with session.post(
            url,
            data=json.dumps(payload).encode(),
            headers={"Accept": "text/event-stream"},
            timeout=aiohttp.ClientTimeout(total=3600),
            proxy=self._proxy,
        ) as resp:
            await self._raise_for_status(resp)
            buf = ""
            async for raw in resp.content:
                buf += raw.decode("utf-8", errors="replace")
                while "\
\
" in buf:
                    block, buf = buf.split("\
\
", 1)
                    event_type: str | None = None
                    data_str:   str | None = None
                    for line in block.splitlines():
                        line = line.strip()
                        if line.startswith("event:"):
                            event_type = line[6:].strip()
                        elif line.startswith("data:"):
                            data_str = line[5:].strip()
                    if event_type and data_str:
                        try:
                            yield {"event": event_type, "data": json.loads(data_str)}
                        except json.JSONDecodeError:
                            pass

    # ── generic feature POST (non-chat) ───────────────────────────────────

    async def _feature_post(self, payload: dict) -> dict:
        """POST to the team features endpoint and return the raw response."""
        await self._get_team_id()  # ensure team_id is resolved
        url = self._team_url(_FEATURE_URL)
        return await self._post(url, payload)

    # ══════════════════════════════════════════════════════════════════════
    # AUTH
    # ══════════════════════════════════════════════════════════════════════

    async def oauth_login(
        self,
        oauth_token: str,
        referrer_id: str | None = None,
        source: str | None = None,
    ) -> UserRecord:
        """
        Authenticate with 1min.AI using a Google OAuth access token.

        Calls ``POST /auth/oauth`` exactly as the browser does.  On success,
        the returned JWT is stored internally and all subsequent requests are
        authenticated automatically.

        Parameters
        ----------
        oauth_token:
            A Google OAuth 2.0 access token (``ya29.…``).
        referrer_id:
            Optional referral ID.
        source:
            Optional referral source string.

        Returns
        -------
        UserRecord
            The authenticated user's profile, including team and credit info.

        Raises
        ------
        OAuthError
            Non-2xx response or missing ``user.token`` in the response body.
        """
        device_id = str(_uuid.uuid4())
        headers   = {
            "Content-Type":  "application/json",
            "Accept":        "application/json, text/plain, */*",
            "X-Auth-Token":  "Bearer",
            "X-App-Version": APP_VERSION,
            "MP-Identity":   f"$device:{device_id}",
            "Origin":        "https://app.1min.ai",
            "Referer":       "https://app.1min.ai/",
        }
        payload = {
            "oauthToken": oauth_token,
            "referral":   {"referrerId": referrer_id, "source": source},
        }

        async with aiohttp.ClientSession() as tmp:
            async with tmp.post(
                _OAUTH_URL,
                data=json.dumps(payload).encode(),
                headers=headers,
                timeout=self._timeout,
            ) as resp:
                body = await resp.json(content_type=None)
                if not resp.ok:
                    raise OAuthError(
                        f"OAuth failed ({resp.status}): {body.get('message', body)}"
                    )

        user_obj = body.get("user")
        if not user_obj:
            raise OAuthError(
                "OAuth response missing 'user' object. "
                f"Response keys: {list(body)}"
            )
        token = user_obj.get("token")
        if not token:
            raise OAuthError(
                f"OAuth user object missing 'token'. Keys: {list(user_obj)}"
            )

        self._api_key = token
        if self._session and not self._session.closed:
            self._session.headers.update({"API-KEY": token})
        else:
            await self.close()

        user = self._parse_user(user_obj)
        self._user    = user
        self._team_id = user.team_id
        logger.info("oauth_login: authenticated as %s", user.email)
        return user

    # ══════════════════════════════════════════════════════════════════════
    # USER & TEAM
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _parse_user(user_obj: dict) -> UserRecord:
        teams     = user_obj.get("teams", [])
        first     = teams[0] if teams else {}
        team_data = first.get("team", {})
        sub       = team_data.get("subscription", {})
        return UserRecord(
            user_id   = user_obj.get("uuid", ""),
            email     = user_obj.get("email", ""),
            team_id   = team_data.get("uuid", ""),
            team_name = team_data.get("name", ""),
            credit    = team_data.get("credit", 0),
            plan      = sub.get("plan", ""),
            settings  = user_obj.get("settings") or {},
            metadata  = user_obj,
        )

    async def get_current_user(self) -> UserRecord:
        """
        Fetch the current authenticated user and primary team.

        Also caches the team ID for all subsequent team-scoped calls.

        Returns
        -------
        UserRecord
        """
        data     = await self._get_json(_USERS_URL)
        user_obj = data.get("user", data)  # type: ignore[union-attr]
        user     = self._parse_user(user_obj)
        self._user    = user
        self._team_id = user.team_id
        return user

    async def update_user_settings(self, **settings: object) -> UserRecord:
        """
        Update user-level settings.

        Keyword arguments are merged into the settings body, e.g.::

            await client.update_user_settings(chatbot=False, homepage="social")

        Returns
        -------
        UserRecord
            The updated user object.
        """
        data     = await self._put(_USER_SETTINGS_URL, dict(settings))
        user_obj = data.get("user", data)
        return self._parse_user(user_obj)

    async def get_team_credits(self) -> int:
        """
        Return the current credit balance of the primary team.

        Returns
        -------
        int
        """
        await self._get_team_id()
        data = await self._get_json(self._team_url(_CREDITS_URL))
        return data.get("credit", 0)  # type: ignore[union-attr]

    async def estimate_chat_cost(
        self,
        prompt: str,
        models: list[str],
        *,
        history_message_limit: int = 8,
        is_mixed: bool = False,
        web_search: bool = False,
        images: list[str] | None = None,
        files: list[str] | None = None,
    ) -> CreditEstimate:
        """
        Estimate credit cost for a chat prompt across one or more models.

        Parameters
        ----------
        prompt:
            The prompt to estimate cost for.
        models:
            List of model ID strings to include in the estimate.
        history_message_limit:
            History context window to assume.
        is_mixed:
            Whether mixed-model history is enabled.
        web_search:
            Whether web search will be used.
        images:
            Image attachment list (affects token count).
        files:
            File attachment list.

        Returns
        -------
        CreditEstimate
        """
        await self._get_team_id()
        payload = {
            "prompt":  prompt,
            "settings": {
                "historySettings": {
                    "historyMessageLimit": history_message_limit,
                    "isMixed":             is_mixed,
                },
                "webSearchSettings": {"webSearch": web_search},
            },
            "attachments": {
                "images": images or [],
                "files":  files  or [],
            },
            "models": models,
        }
        data  = await self._post(self._team_url(_ESTIMATE_URL), payload)
        total = data.get("total", {})
        return CreditEstimate(
            models                 = data.get("models", {}),
            total_input_tokens     = total.get("inputTokens", 0),
            total_estimated_credit = total.get("estimatedCredit", 0),
        )

    async def list_team_members(self) -> list[dict]:
        """Return all members of the primary team."""
        await self._get_team_id()
        data = await self._get_json(self._team_url(_MEMBERS_URL))
        return data.get("members", [])  # type: ignore[union-attr]

    async def get_notebook(self) -> str | None:
        """Return the user's notebook content, or ``None`` if empty."""
        data = await self._get_json(_NOTEBOOK_URL)
        return data.get("notebook")  # type: ignore[union-attr]

    # ══════════════════════════════════════════════════════════════════════
    # MODELS CATALOG
    # ══════════════════════════════════════════════════════════════════════

    async def list_models(
        self,
        feature: str | None = None,
    ) -> list[dict]:
        """
        Return the full model catalog from ``GET /models``.

        Parameters
        ----------
        feature:
            If provided, filter to models that support this feature,
            e.g. ``"UNIFY_CHAT_WITH_AI"``, ``"IMAGE_GENERATOR"``.

        Returns
        -------
        list[dict]
            Each dict contains ``modelId``, ``name``, ``provider``,
            ``features``, ``creditMetadata``, ``modality``, etc.
        """
        data   = await self._get_json(_MODELS_URL)
        models = data.get("models", [])  # type: ignore[union-attr]
        if feature:
            models = [m for m in models if feature in m.get("features", [])]
        return models

    # ══════════════════════════════════════════════════════════════════════
    # CHAT
    # ══════════════════════════════════════════════════════════════════════

    def _build_chat_payload(
        self,
        prompt: str,
        model: str,
        conversation_id: str | None,
        images: list[str] | None,
        files: list[str] | None,
        web_search: bool,
        num_of_site: int,
        max_word: int,
        is_mixed: bool,
        history_message_limit: int,
        with_memories: bool,
        brand_voice_id: str | None,
    ) -> dict:
        payload: dict = {
            "type":           "UNIFY_CHAT_WITH_AI",
            "model":          model,
            "promptObject": {
                "prompt":         prompt,
                "conversationId": conversation_id or "",
                "settings": {
                    "webSearchSettings": {
                        "webSearch":  web_search,
                        "numOfSite":  num_of_site,
                        "maxWord":    max_word,
                    },
                    "historySettings": {
                        "isMixed":             is_mixed,
                        "historyMessageLimit": history_message_limit,
                    },
                    "withMemories": with_memories,
                },
                "attachments": {
                    "images": images or [],
                    "files":  files  or [],
                },
            },
            "metadata": {
                "messageGroup": _message_group(),
            },
        }
        if conversation_id:
            payload["conversationId"] = conversation_id
        if brand_voice_id:
            payload["brandVoiceId"] = brand_voice_id
        return payload

    @staticmethod
    def _parse_chat_response(data: dict) -> ChatOutput:
        record  = data.get("aiRecord", {})
        detail  = record.get("aiRecordDetail", {})
        results = detail.get("resultObject", [])
        text    = results[0] if isinstance(results, list) and results else ""
        return ChatOutput(
            text            = text,
            model           = record.get("model", ""),
            conversation_id = record.get("conversationId"),
            record_id       = record.get("uuid"),
            metadata        = record,
        )

    @overload
    async def chat(
        self,
        prompt: str,
        *,
        stream: Literal[False] = ...,
        model: str | ChatModel | None = ...,
        conversation_id: str | None = ...,
        images: list[str] | None = ...,
        files: list[str] | None = ...,
        web_search: bool = ...,
        num_of_site: int = ...,
        max_word: int = ...,
        is_mixed: bool = ...,
        history_message_limit: int = ...,
        with_memories: bool = ...,
        brand_voice_id: str | None = ...,
    ) -> ChatOutput: ...

    @overload
    async def chat(
        self,
        prompt: str,
        *,
        stream: Literal[True],
        model: str | ChatModel | None = ...,
        conversation_id: str | None = ...,
        images: list[str] | None = ...,
        files: list[str] | None = ...,
        web_search: bool = ...,
        num_of_site: int = ...,
        max_word: int = ...,
        is_mixed: bool = ...,
        history_message_limit: int = ...,
        with_memories: bool = ...,
        brand_voice_id: str | None = ...,
    ) -> AsyncIterator[ChatOutput]: ...

    async def chat(
        self,
        prompt: str,
        *,
        stream: bool = False,
        model: str | ChatModel | None = None,
        conversation_id: str | None = None,
        images: list[str] | None = None,
        files: list[str] | None = None,
        web_search: bool = False,
        num_of_site: int = 2,
        max_word: int = 1000,
        is_mixed: bool = False,
        history_message_limit: int = 8,
        with_memories: bool = False,
        brand_voice_id: str | None = None,
    ) -> ChatOutput | AsyncIterator[ChatOutput]:
        """
        Send a chat message to a 1min.AI model.

        Parameters
        ----------
        prompt:
            The user message.
        stream:
            When ``True``, returns an async iterator of incremental
            :class:`~1minai.types.ChatOutput` chunks.
        model:
            Model to use.  Defaults to ``gpt-4.1-nano``.
        conversation_id:
            Continue an existing server-side thread.
        images:
            Image URLs or asset keys to attach.
        files:
            File UUIDs from the Asset API to attach (PDFs, documents, etc.).
        web_search:
            Enable live web search grounding.
        num_of_site:
            Number of websites to search (when ``web_search=True``).
        max_word:
            Maximum word count pulled from web results.
        is_mixed:
            Allow mixing context from different models in history.
        history_message_limit:
            How many past messages to include as context.
        with_memories:
            Enable AI memory across sessions.
        brand_voice_id:
            Optional brand voice UUID for a custom output style.

        Returns
        -------
        ChatOutput
            When ``stream=False`` (default).
        AsyncIterator[ChatOutput]
            When ``stream=True``.

        Example::

            r = await client.chat("What is the capital of Poland?")
            print(r.text)

            async for chunk in await client.chat("Write a poem", stream=True):
                print(chunk.text_delta, end="", flush=True)
        """
        await self._get_team_id()
        resolved = _resolve(model, DEFAULT_CHAT_MODEL)
        payload  = self._build_chat_payload(
            prompt, resolved, conversation_id, images, files,
            web_search, num_of_site, max_word, is_mixed,
            history_message_limit, with_memories, brand_voice_id,
        )
        url = self._team_url(_CHAT_V2_URL)

        if stream:
            return self._chat_stream(url, payload)

        data = await self._post(url, payload)
        return self._parse_chat_response(data)

    async def _chat_stream(
        self, url: str, payload: dict
    ) -> AsyncIterator[ChatOutput]:
        accumulated = ""
        async for evt in self._post_stream(url, payload):
            event = evt["event"]
            data  = evt["data"]
            if event == "content":
                delta        = data.get("content", "")
                accumulated += delta
                yield ChatOutput(
                    text       = accumulated,
                    text_delta = delta,
                    metadata   = {},
                )
            elif event == "result":
                record = data.get("aiRecord", {})
                yield ChatOutput(
                    text            = accumulated,
                    text_delta      = "",
                    model           = record.get("model", ""),
                    conversation_id = record.get("conversationId"),
                    record_id       = record.get("uuid"),
                    metadata        = record,
                )

    # ── single-turn convenience ────────────────────────────────────────────

    async def generate_content(
        self,
        prompt: str,
        model: str | ChatModel | None = None,
        *,
        images: list[str] | None = None,
        files: list[str] | None = None,
        web_search: bool = False,
        brand_voice_id: str | None = None,
    ) -> ChatOutput:
        """
        Single-turn message; returns the full response.

        Example::

            r = await client.generate_content("Translate 'hello' to Japanese.")
            print(r.text)
        """
        return await self.chat(
            prompt,
            stream         = False,
            model          = model,
            images         = images,
            files          = files,
            web_search     = web_search,
            brand_voice_id = brand_voice_id,
        )

    async def generate_content_stream(
        self,
        prompt: str,
        model: str | ChatModel | None = None,
        *,
        images: list[str] | None = None,
        files: list[str] | None = None,
        web_search: bool = False,
        brand_voice_id: str | None = None,
    ) -> AsyncIterator[ChatOutput]:
        """
        Stream a single-turn response.

        Example::

            async for chunk in client.generate_content_stream("Tell me a story"):
                print(chunk.text_delta, end="", flush=True)
        """
        async for chunk in await self.chat(
            prompt,
            stream         = True,
            model          = model,
            images         = images,
            files          = files,
            web_search     = web_search,
            brand_voice_id = brand_voice_id,
        ):
            yield chunk

    # ── multi-turn session ────────────────────────────────────────────────

    def start_chat(
        self,
        model: str | ChatModel | None = None,
        *,
        conversation_id: str | None = None,
        web_search: bool = False,
        num_of_site: int = 2,
        max_word: int = 1000,
        is_mixed: bool = False,
        history_message_limit: int = 8,
        with_memories: bool = False,
        brand_voice_id: str | None = None,
    ) -> ChatSession:
        """
        Create a :class:`~1minai.session.ChatSession` for multi-turn chat.

        Example::

            chat = client.start_chat(model="gpt-4.1-nano")
            r1   = await chat.send_message("My name is Alice.")
            r2   = await chat.send_message("What's my name?")
            print(r2.text)
        """
        return ChatSession(
            client                = self,
            model                 = _resolve(model, DEFAULT_CHAT_MODEL),
            conversation_id       = conversation_id,
            web_search            = web_search,
            num_of_site           = num_of_site,
            max_word              = max_word,
            is_mixed              = is_mixed,
            history_message_limit = history_message_limit,
            with_memories         = with_memories,
            brand_voice_id        = brand_voice_id,
        )

    # ══════════════════════════════════════════════════════════════════════
    # CONVERSATIONS
    # ══════════════════════════════════════════════════════════════════════

    async def create_conversation(
        self,
        title: str,
    ) -> ConversationRecord:
        """
        Create a server-side conversation for persistent history.

        Pass the returned ``conversation_id`` to :meth:`chat` or
        :meth:`start_chat` to continue the thread.

        Example::

            conv  = await client.create_conversation("Research session")
            reply = await client.chat("What is quantum entanglement?",
                                      conversation_id=conv.conversation_id)
        """
        await self._get_team_id()
        data = await self._post(
            self._team_url(_CONV_URL),
            {"type": "UNIFY_CHAT_WITH_AI", "title": title},
        )
        return ConversationRecord(
            conversation_id = data.get("uuid", ""),
            title           = data.get("title", title),
            metadata        = data,
        )

    async def list_conversations(self) -> list[ConversationRecord]:
        """Return all server-side conversations for the current team."""
        await self._get_team_id()
        data  = await self._get_json(self._team_url(_CONV_URL))
        items = data.get("conversationList", [])  # type: ignore[union-attr]
        return [
            ConversationRecord(
                conversation_id = c.get("uuid", ""),
                title           = c.get("title", ""),
                metadata        = c,
            )
            for c in items
        ]

    async def get_conversation(self, conversation_id: str) -> ConversationRecord:
        """Fetch a single conversation by its UUID."""
        await self._get_team_id()
        url  = self._team_url(_CONV_DETAIL_URL, conv_id=conversation_id)
        data = await self._get_json(url)
        c    = data.get("conversation", data)  # type: ignore[union-attr]
        return ConversationRecord(
            conversation_id = c.get("uuid", conversation_id),
            title           = c.get("title", ""),
            metadata        = c,
        )

    async def get_conversation_messages(
        self,
        conversation_id: str,
        after_id: str | None = None,
    ) -> list[MessageRecord]:
        """
        Return the message history of a conversation.

        Parameters
        ----------
        conversation_id:
            UUID of the conversation.
        after_id:
            Pagination cursor — return only messages after this record UUID.

        Returns
        -------
        list[MessageRecord]
        """
        await self._get_team_id()
        params: dict = {"conversationId": conversation_id}
        if after_id:
            params["afterId"] = after_id
        data  = await self._get_json(
            self._team_url(_MSG_URL), params=params
        )
        items = data.get("messageList", [])  # type: ignore[union-attr]
        return [
            MessageRecord(
                role           = m.get("role", ""),
                content        = m.get("content", ""),
                record_id      = m.get("aiRecordId", ""),
                credit         = m.get("credit", 0),
                execution_time = m.get("executionTime", 0.0),
                metadata       = m,
            )
            for m in items
        ]

    async def delete_conversation(self, conversation_id: str) -> None:
        """Delete *conversation_id* from the server."""
        await self._get_team_id()
        url = self._team_url(_CONV_DETAIL_URL, conv_id=conversation_id)
        await self._delete(url)
        logger.info("Deleted conversation %s…", conversation_id[:8])

    # ══════════════════════════════════════════════════════════════════════
    # FEATURE SETTINGS
    # ══════════════════════════════════════════════════════════════════════

    async def get_feature_settings(
        self, feature: str = "UNIFY_CHAT_WITH_AI"
    ) -> dict:
        """
        Return the last-used settings for a feature (e.g. preferred models).

        Parameters
        ----------
        feature:
            Feature type string, e.g. ``"UNIFY_CHAT_WITH_AI"``.

        Returns
        -------
        dict
            Raw feature-settings object including ``lastInputSubmission``.
        """
        await self._get_team_id()
        return await self._get_json(  # type: ignore[return-value]
            self._team_url(_SETTINGS_URL, feature=feature)
        )

    async def update_feature_settings(
        self, models: list[str], feature: str = "UNIFY_CHAT_WITH_AI"
    ) -> dict:
        """
        Persist the preferred model list for a feature.

        Parameters
        ----------
        models:
            List of model ID strings to store, e.g. ``["gpt-4.1-nano"]``.
        feature:
            Feature type string.

        Returns
        -------
        dict
            Raw response from the server.
        """
        await self._get_team_id()
        return await self._post(
            self._team_url(_SETTINGS_URL, feature=feature),
            {"models": models},
        )

    # ══════════════════════════════════════════════════════════════════════
    # NOTIFICATIONS
    # ══════════════════════════════════════════════════════════════════════

    async def get_unread_notification_count(self) -> int:
        """Return the number of unread notifications."""
        data = await self._get_json(_NOTIF_UNREAD_URL)
        return data.get("unread", 0)  # type: ignore[union-attr]

    async def get_notification(self, notification_id: str) -> NotificationRecord:
        """Fetch a single notification by its UUID."""
        url  = _NOTIF_DETAIL_URL.format(notif_id=notification_id)
        data = await self._get_json(url)
        n    = data.get("notification", data)  # type: ignore[union-attr]
        return NotificationRecord(
            notification_id = n.get("uuid", ""),
            type            = n.get("type", ""),
            scope           = n.get("scope", ""),
            content         = n.get("content", ""),
            status          = n.get("status", ""),
            metadata        = n,
        )

    # ══════════════════════════════════════════════════════════════════════
    # TEXT TOOLS
    # ══════════════════════════════════════════════════════════════════════

    async def _text_tool(
        self,
        feature_type: str,
        prompt_obj: dict,
        model: str,
    ) -> ChatOutput:
        """Generic helper for single-text-input/output feature calls."""
        payload = {
            "type":         feature_type,
            "model":        model,
            "promptObject": prompt_obj,
        }
        data   = await self._feature_post(payload)
        record = data.get("aiRecord", {})
        return ChatOutput(
            text      = _text_from_record(record),
            model     = record.get("model", model),
            record_id = record.get("uuid"),
            metadata  = record,
        )

    async def generate_code(
        self,
        prompt: str,
        model: str | CodeModel | None = None,
        *,
        web_search: bool = False,
    ) -> ChatOutput:
        """
        Generate code using a specialised coding model.

        Example::

            r = await client.generate_code(
                "Write a Python async HTTP client using aiohttp."
            )
            print(r.text)
        """
        resolved = _resolve(model, DEFAULT_CODE_MODEL)
        payload  = {
            "type":           "CODE_GENERATOR",
            "model":          resolved,
            "conversationId": "CODE_GENERATOR",
            "promptObject":   {"prompt": prompt, "webSearch": web_search},
        }
        data   = await self._feature_post(payload)
        record = data.get("aiRecord", {})
        return ChatOutput(
            text      = _text_from_record(record),
            model     = record.get("model", resolved),
            record_id = record.get("uuid"),
            metadata  = record,
        )

    async def check_grammar(
        self,
        text: str,
        model: str | ChatModel | None = None,
    ) -> ChatOutput:
        """
        Check and correct grammar in *text*.

        Returns a :class:`ChatOutput` whose ``text`` is the corrected version.
        """
        return await self._text_tool(
            "GRAMMAR_CHECKER",
            {"prompt": text},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def paraphrase(
        self,
        text: str,
        model: str | ChatModel | None = None,
        *,
        tone: str = "professional",
        language: str = "English",
    ) -> ChatOutput:
        """Paraphrase *text* with the given tone and language."""
        return await self._text_tool(
            "PARAPHRASER",
            {"prompt": text, "tone": tone, "language": language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def rewrite(
        self,
        text: str,
        model: str | ChatModel | None = None,
        *,
        tone: str = "professional",
        language: str = "English",
    ) -> ChatOutput:
        """Rewrite *text* with the given tone and language."""
        return await self._text_tool(
            "REWRITER",
            {"prompt": text, "tone": tone, "language": language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def summarize(
        self,
        text: str,
        model: str | ChatModel | None = None,
        *,
        language: str = "English",
    ) -> ChatOutput:
        """Summarize *text*."""
        return await self._text_tool(
            "SUMMARIZER",
            {"prompt": text, "language": language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def expand_content(
        self,
        text: str,
        model: str | ChatModel | None = None,
        *,
        language: str = "English",
    ) -> ChatOutput:
        """Expand short *text* into a longer version."""
        return await self._text_tool(
            "CONTENT_EXPANDER",
            {"prompt": text, "language": language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def shorten_content(
        self,
        text: str,
        model: str | ChatModel | None = None,
        *,
        language: str = "English",
    ) -> ChatOutput:
        """Shorten *text* while preserving meaning."""
        return await self._text_tool(
            "CONTENT_SHORTENER",
            {"prompt": text, "language": language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def translate(
        self,
        text: str,
        target_language: str,
        model: str | ChatModel | None = None,
    ) -> ChatOutput:
        """
        Translate *text* into *target_language*.

        Example::

            r = await client.translate("Good morning", "Polish")
            print(r.text)   # "Dzień dobry"
        """
        return await self._text_tool(
            "CONTENT_TRANSLATOR",
            {"prompt": text, "language": target_language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def detect_ai_content(
        self,
        text: str,
    ) -> ChatOutput:
        """
        Detect whether *text* was written by AI (uses Winston AI).

        Returns
        -------
        ChatOutput
            ``text`` contains the detection report.
        """
        return await self._text_tool(
            "CONTENT_DETECTOR",
            {"prompt": text},
            "winstonai",
        )

    async def research_keywords(
        self,
        topic: str,
        model: str | ChatModel | None = None,
    ) -> ChatOutput:
        """Generate SEO keyword research for *topic*."""
        return await self._text_tool(
            "KEYWORD_RESEARCH",
            {"prompt": topic},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    # ── content generators ────────────────────────────────────────────────

    async def generate_blog_article(
        self,
        prompt: str,
        model: str | ChatModel | None = None,
        *,
        tone: str = "professional",
        language: str = "English",
    ) -> ChatOutput:
        """Generate a blog article from *prompt*."""
        return await self._text_tool(
            "CONTENT_GENERATOR_BLOG_ARTICLE",
            {"prompt": prompt, "tone": tone, "language": language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def generate_email(
        self,
        prompt: str,
        model: str | ChatModel | None = None,
        *,
        tone: str = "professional",
        language: str = "English",
    ) -> ChatOutput:
        """Generate an email from *prompt*."""
        return await self._text_tool(
            "CONTENT_GENERATOR_EMAIL",
            {"prompt": prompt, "tone": tone, "language": language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def generate_email_reply(
        self,
        original_email: str,
        instructions: str,
        model: str | ChatModel | None = None,
        *,
        tone: str = "professional",
        language: str = "English",
    ) -> ChatOutput:
        """Generate a reply to *original_email* following *instructions*."""
        return await self._text_tool(
            "CONTENT_GENERATOR_EMAIL_REPLY",
            {
                "prompt":        instructions,
                "originalEmail": original_email,
                "tone":          tone,
                "language":      language,
            },
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def generate_social_post(
        self,
        prompt: str,
        platform: str,
        model: str | ChatModel | None = None,
        *,
        tone: str = "casual",
        language: str = "English",
    ) -> ChatOutput:
        """
        Generate a social media post for *platform*.

        Parameters
        ----------
        prompt:
            Topic or brief for the post.
        platform:
            One of: ``"instagram"``, ``"facebook"``, ``"linkedin"``,
            ``"tiktok"``, ``"x"`` (Twitter/X).  A ``"_post"`` / ``"_description"``
            suffix is appended automatically for the correct feature type.
        tone:
            Writing tone.
        language:
            Output language.

        Returns
        -------
        ChatOutput
        """
        _PLATFORM_MAP = {
            "instagram": "CONTENT_GENERATOR_INSTAGRAM_POST",
            "facebook":  "CONTENT_GENERATOR_FACEBOOK_POST",
            "linkedin":  "CONTENT_GENERATOR_LINKEDIN_POST",
            "tiktok":    "CONTENT_GENERATOR_TIKTOK_DESCRIPTION",
            "x":         "CONTENT_GENERATOR_X_TWEET",
            "twitter":   "CONTENT_GENERATOR_X_TWEET",
        }
        feature = _PLATFORM_MAP.get(platform.lower())
        if not feature:
            raise ValueError(
                f"Unknown platform {platform!r}.  "
                f"Choose from: {list(_PLATFORM_MAP)}"
            )
        return await self._text_tool(
            feature,
            {"prompt": prompt, "tone": tone, "language": language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def generate_social_comment(
        self,
        post_text: str,
        platform: str,
        model: str | ChatModel | None = None,
        *,
        tone: str = "casual",
        language: str = "English",
    ) -> ChatOutput:
        """
        Generate a comment reply for a social media post.

        Parameters
        ----------
        post_text:
            The post to reply to.
        platform:
            ``"facebook"``, ``"linkedin"``, or ``"x"`` (Twitter/X).
        """
        _COMMENT_MAP = {
            "facebook": "CONTENT_GENERATOR_FACEBOOK_COMMENT",
            "linkedin": "CONTENT_GENERATOR_LINKEDIN_COMMENT",
            "x":        "CONTENT_GENERATOR_X_COMMENT",
            "twitter":  "CONTENT_GENERATOR_X_COMMENT",
        }
        feature = _COMMENT_MAP.get(platform.lower())
        if not feature:
            raise ValueError(
                f"Unknown platform {platform!r}.  "
                f"Choose from: {list(_COMMENT_MAP)}"
            )
        return await self._text_tool(
            feature,
            {"prompt": post_text, "tone": tone, "language": language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def generate_ad_copy(
        self,
        prompt: str,
        platform: str,
        model: str | ChatModel | None = None,
        *,
        tone: str = "persuasive",
        language: str = "English",
    ) -> ChatOutput:
        """
        Generate advertising copy.

        Parameters
        ----------
        platform:
            ``"facebook"`` or ``"google"``.
        """
        _AD_MAP = {
            "facebook": "CONTENT_GENERATOR_FACEBOOK_ADS",
            "google":   "CONTENT_GENERATOR_GOOGLE_ADS",
        }
        feature = _AD_MAP.get(platform.lower())
        if not feature:
            raise ValueError(
                f"Unknown ad platform {platform!r}. Choose from: {list(_AD_MAP)}"
            )
        return await self._text_tool(
            feature,
            {"prompt": prompt, "tone": tone, "language": language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def generate_presentation(
        self,
        prompt: str,
        model: str | ChatModel | None = None,
        *,
        language: str = "English",
    ) -> ChatOutput:
        """Generate a presentation outline or slide content from *prompt*."""
        return await self._text_tool(
            "PRESENTATION_GENERATOR",
            {"prompt": prompt, "language": language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    # ── YouTube tools ─────────────────────────────────────────────────────

    async def summarize_youtube(
        self,
        youtube_url: str,
        model: str | ChatModel | None = None,
        *,
        language: str = "English",
    ) -> ChatOutput:
        """
        Summarize a YouTube video.

        Parameters
        ----------
        youtube_url:
            Full YouTube video URL.
        """
        return await self._text_tool(
            "YOUTUBE_SUMMARIZER",
            {"prompt": youtube_url, "language": language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def transcribe_youtube(
        self,
        youtube_url: str,
        model: str | ChatModel | None = None,
    ) -> ChatOutput:
        """Return the full transcript of a YouTube video."""
        return await self._text_tool(
            "YOUTUBE_TRANSCRIBER",
            {"prompt": youtube_url},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def translate_youtube(
        self,
        youtube_url: str,
        target_language: str,
        model: str | ChatModel | None = None,
    ) -> ChatOutput:
        """
        Translate the transcript of a YouTube video.

        Parameters
        ----------
        youtube_url:
            Full YouTube video URL.
        target_language:
            Language to translate into, e.g. ``"Polish"``.
        """
        return await self._text_tool(
            "YOUTUBE_TRANSLATOR",
            {"prompt": youtube_url, "language": target_language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    async def translate_document(
        self,
        file_id: str,
        target_language: str,
        model: str | ChatModel | None = None,
    ) -> ChatOutput:
        """
        Translate an uploaded document.

        Parameters
        ----------
        file_id:
            UUID from :meth:`upload_asset` (``asset.file_id``).
        target_language:
            Target language, e.g. ``"Spanish"``.
        """
        return await self._text_tool(
            "DOCUMENT_TRANSLATOR",
            {"fileId": file_id, "language": target_language},
            _resolve(model, DEFAULT_CHAT_MODEL),
        )

    # ══════════════════════════════════════════════════════════════════════
    # IMAGE GENERATION & EDITING
    # ══════════════════════════════════════════════════════════════════════

    async def generate_image(
        self,
        prompt: str,
        model: str | ImageModel | None = None,
        *,
        negative_prompt: str | None = None,
        width: int = 1024,
        height: int = 1024,
        num_images: int = 1,
        quality: str = "standard",
        style: str = "vivid",
        output_format: str = "webp",
    ) -> ImageOutput:
        """
        Generate images from a text prompt.

        Parameters
        ----------
        prompt:
            Detailed image description.
        model:
            Image model.  Defaults to ``flux-1.1-pro``.
        negative_prompt:
            What to avoid in the output.
        width / height:
            Output dimensions in pixels.
        num_images:
            Number of images (1–4).
        quality:
            ``"standard"``, ``"hd"``, or ``"ultra"``.
        style:
            ``"vivid"``, ``"natural"``, or ``"cinematic"``.
        output_format:
            Output file format, e.g. ``"webp"``, ``"png"``, ``"jpeg"``.

        Returns
        -------
        ImageOutput

        Example::

            result = await client.generate_image(
                "A golden sunset over Warsaw, oil painting style.",
            )
            await result.image.save(".")
        """
        resolved   = _resolve(model, DEFAULT_IMAGE_MODEL)
        prompt_obj: dict = {
            "prompt":       prompt,
            "width":        width,
            "height":       height,
            "numImages":    num_images,
            "quality":      quality,
            "style":        style,
            "output_format": output_format,
        }
        if negative_prompt:
            prompt_obj["negativePrompt"] = negative_prompt

        data   = await self._feature_post({
            "type":         "IMAGE_GENERATOR",
            "model":        resolved,
            "promptObject": prompt_obj,
        })
        record = data.get("aiRecord", {})
        urls   = _extract_result_urls(record)
        return ImageOutput(
            images    = [GeneratedImage(url=u, model=resolved) for u in urls],
            model     = resolved,
            record_id = record.get("uuid"),
            metadata  = record,
        )

    def _image_output(self, data: dict, model: str = "") -> ImageOutput:
        record = data.get("aiRecord", {})
        urls   = _extract_result_urls(record)
        return ImageOutput(
            images    = [GeneratedImage(url=u, model=model) for u in urls],
            model     = model,
            record_id = record.get("uuid"),
            metadata  = record,
        )

    async def edit_image(
        self,
        image_url: str,
        prompt: str,
        model: str | ImageModel | None = None,
    ) -> ImageOutput:
        """
        Edit an existing image with a text instruction (``IMAGE_EDITOR``).

        Supports Flux Kontext models, Qwen Image Edit, and depth/canny
        control-net models.

        Example::

            r = await client.edit_image(
                "https://example.com/photo.jpg",
                "Make the sky a dramatic purple at sunset.",
                model="black-forest-labs/flux-kontext-pro",
            )
        """
        resolved = _resolve(model, "black-forest-labs/flux-kontext-pro")
        return self._image_output(
            await self._feature_post({
                "type":         "IMAGE_EDITOR",
                "model":        resolved,
                "promptObject": {"imageList": [image_url], "prompt": prompt},
            }),
            resolved,
        )

    async def create_image_variation(
        self,
        image_url: str,
        model: str | ImageModel | None = None,
        *,
        num_images: int = 1,
    ) -> ImageOutput:
        """
        Create variations of an existing image (``IMAGE_VARIATOR``).

        Supports Flux Redux, Dzine, Magic Art.
        """
        resolved = _resolve(model, "black-forest-labs/flux-redux-dev")
        return self._image_output(
            await self._feature_post({
                "type":         "IMAGE_VARIATOR",
                "model":        resolved,
                "promptObject": {"imageList": [image_url], "numImages": num_images},
            }),
            resolved,
        )

    async def extend_image(
        self,
        image_url: str,
        direction: str = "right",
        model: str | ImageModel | None = None,
    ) -> ImageOutput:
        """
        Extend / outpaint an image beyond its edges (``IMAGE_EXTENDER``).

        Parameters
        ----------
        image_url:
            URL or asset key of the source image.
        direction:
            Direction to extend: ``"left"``, ``"right"``, ``"up"``, ``"down"``.
        model:
            Defaults to ``stable-image`` (Stable Image Core).
        """
        resolved = _resolve(model, "stable-image")
        return self._image_output(
            await self._feature_post({
                "type":         "IMAGE_EXTENDER",
                "model":        resolved,
                "promptObject": {"imageList": [image_url], "direction": direction},
            }),
            resolved,
        )

    async def inpaint_image(
        self,
        image_url: str,
        mask_url: str,
        prompt: str,
        model: str | ImageModel | None = None,
    ) -> ImageOutput:
        """
        Inpaint a masked region of an image (``IMAGE_INPAINTER``).

        Parameters
        ----------
        image_url:
            Source image URL or asset key.
        mask_url:
            Mask image URL or asset key (white = region to fill).
        prompt:
            Text description of what to place in the masked region.
        """
        resolved = _resolve(model, "black-forest-labs/flux-fill-pro")
        return self._image_output(
            await self._feature_post({
                "type":         "IMAGE_INPAINTER",
                "model":        resolved,
                "promptObject": {
                    "imageList": [image_url],
                    "maskList":  [mask_url],
                    "prompt":    prompt,
                },
            }),
            resolved,
        )

    async def remove_object_from_image(
        self,
        image_url: str,
        model: str | ImageModel | None = None,
    ) -> ImageOutput:
        """Remove objects from an image (``IMAGE_OBJECT_REMOVER``)."""
        resolved = _resolve(model, "recraft")
        return self._image_output(
            await self._feature_post({
                "type":         "IMAGE_OBJECT_REMOVER",
                "model":        resolved,
                "promptObject": {"imageList": [image_url]},
            }),
            resolved,
        )

    async def upscale_image(
        self,
        image_url: str,
        scale: int = 2,
        model: str | ImageModel | None = None,
    ) -> ImageOutput:
        """
        Upscale an image (``IMAGE_UPSCALER``).

        Parameters
        ----------
        scale:
            Upscale factor: ``2`` or ``4``.
        """
        resolved = _resolve(model, "stable-image")
        return self._image_output(
            await self._feature_post({
                "type":         "IMAGE_UPSCALER",
                "model":        resolved,
                "promptObject": {"imageList": [image_url], "scale": scale},
            }),
            resolved,
        )

    async def remove_background(
        self,
        image_url: str,
        model: str | ImageModel | None = None,
    ) -> ImageOutput:
        """Remove the background from an image (``BACKGROUND_REMOVER``)."""
        resolved = _resolve(model, "clipdrop")
        return self._image_output(
            await self._feature_post({
                "type":         "BACKGROUND_REMOVER",
                "model":        resolved,
                "promptObject": {"imageList": [image_url]},
            }),
            resolved,
        )

    async def replace_background(
        self,
        image_url: str,
        prompt: str,
        model: str | ImageModel | None = None,
    ) -> ImageOutput:
        """
        Replace the background of an image using a text prompt
        (``BACKGROUND_REPLACER``).
        """
        resolved = _resolve(model, "gpt-image-1")
        return self._image_output(
            await self._feature_post({
                "type":         "BACKGROUND_REPLACER",
                "model":        resolved,
                "promptObject": {"imageList": [image_url], "prompt": prompt},
            }),
            resolved,
        )

    async def remove_text_from_image(self, image_url: str) -> ImageOutput:
        """Remove text overlays from an image (``TEXT_REMOVER``)."""
        return self._image_output(
            await self._feature_post({
                "type":         "TEXT_REMOVER",
                "model":        "clipdrop",
                "promptObject": {"imageList": [image_url]},
            }),
            "clipdrop",
        )

    async def swap_face(
        self,
        source_image_url: str,
        target_image_url: str,
        model: str | ImageModel | None = None,
    ) -> ImageOutput:
        """
        Swap a face from *source_image_url* into *target_image_url*
        (``FACE_SWAPPER``).
        """
        resolved = _resolve(model, "Qubico/image-toolkit")
        return self._image_output(
            await self._feature_post({
                "type":         "FACE_SWAPPER",
                "model":        resolved,
                "promptObject": {
                    "sourceImageList": [source_image_url],
                    "targetImageList": [target_image_url],
                },
            }),
            resolved,
        )

    async def sketch_to_image(
        self,
        sketch_url: str,
        prompt: str,
        model: str | ImageModel | None = None,
    ) -> ImageOutput:
        """
        Convert a sketch/line-art image to a full image (``SKETCH_TO_IMAGE``).
        """
        resolved = _resolve(model, "stable-image")
        return self._image_output(
            await self._feature_post({
                "type":         "SKETCH_TO_IMAGE",
                "model":        resolved,
                "promptObject": {"imageList": [sketch_url], "prompt": prompt},
            }),
            resolved,
        )

    async def image_to_3d(
        self,
        image_url: str,
        model: str | ImageModel | None = None,
    ) -> ImageOutput:
        """
        Convert an image to a 3-D model / representation
        (``IMAGE_3D_GENERATOR``).
        """
        resolved = _resolve(model, "Qubico/trellis")
        return self._image_output(
            await self._feature_post({
                "type":         "IMAGE_3D_GENERATOR",
                "model":        resolved,
                "promptObject": {"imageList": [image_url]},
            }),
            resolved,
        )

    async def image_to_prompt(
        self,
        image_url: str,
        model: str | None = None,
    ) -> ChatOutput:
        """
        Reverse-engineer a text prompt from an image (``IMAGE_TO_PROMPT``).
        """
        resolved = model or "methexis-inc/img2prompt:50adaf2d3ad20a6f911a8a9e3ccf777b263b8596fbd2c8fc26e8888f8a0edbb5"
        data     = await self._feature_post({
            "type":         "IMAGE_TO_PROMPT",
            "model":        resolved,
            "promptObject": {"imageList": [image_url]},
        })
        record = data.get("aiRecord", {})
        return ChatOutput(
            text      = _text_from_record(record),
            model     = record.get("model", resolved),
            record_id = record.get("uuid"),
            metadata  = record,
        )

    async def search_and_replace_in_image(
        self,
        image_url: str,
        search_prompt: str,
        replace_prompt: str,
        model: str | None = None,
    ) -> ImageOutput:
        """
        Search for an object in *image_url* and replace it
        (``SEARCH_AND_REPLACE``).
        """
        resolved = model or "stable-image"
        return self._image_output(
            await self._feature_post({
                "type":         "SEARCH_AND_REPLACE",
                "model":        resolved,
                "promptObject": {
                    "imageList":     [image_url],
                    "searchPrompt":  search_prompt,
                    "replacePrompt": replace_prompt,
                },
            }),
            resolved,
        )

    # ══════════════════════════════════════════════════════════════════════
    # VIDEO
    # ══════════════════════════════════════════════════════════════════════

    async def generate_video(
        self,
        prompt: str,
        model: str | VideoModel | None = None,
        *,
        duration: int = 5,
        aspect_ratio: str = "16:9",
    ) -> VideoOutput:
        """
        Generate a video from a text prompt (``TEXT_TO_VIDEO``).

        Parameters
        ----------
        prompt:
            Description of the video.
        model:
            Defaults to ``kling``.
        duration:
            Length in seconds: ``5`` or ``10``.
        aspect_ratio:
            ``"16:9"``, ``"9:16"``, ``"1:1"``, or ``"4:3"``.

        Example::

            video = await client.generate_video(
                "A drone shot flying over Warsaw at sunset.",
            )
            print(video.video_url)
        """
        resolved = _resolve(model, DEFAULT_VIDEO_MODEL)
        data     = await self._feature_post({
            "type":         "TEXT_TO_VIDEO",
            "model":        resolved,
            "promptObject": {
                "prompt":      prompt,
                "duration":    duration,
                "aspectRatio": aspect_ratio,
            },
        })
        record = data.get("aiRecord", {})
        urls   = _extract_result_urls(record)
        return VideoOutput(
            video_url = urls[0] if urls else "",
            model     = record.get("model", resolved),
            record_id = record.get("uuid"),
            metadata  = record,
        )

    async def image_to_video(
        self,
        image_url: str,
        prompt: str = "",
        model: str | VideoModel | None = None,
        *,
        duration: int = 5,
        aspect_ratio: str = "16:9",
    ) -> VideoOutput:
        """
        Animate a still image into a video (``IMAGE_TO_VIDEO``).

        Parameters
        ----------
        image_url:
            URL or asset key of the source image.
        prompt:
            Optional motion description.
        model:
            Defaults to ``kling``.
        """
        resolved = _resolve(model, DEFAULT_VIDEO_MODEL)
        data     = await self._feature_post({
            "type":         "IMAGE_TO_VIDEO",
            "model":        resolved,
            "promptObject": {
                "imageList":   [image_url],
                "prompt":      prompt,
                "duration":    duration,
                "aspectRatio": aspect_ratio,
            },
        })
        record = data.get("aiRecord", {})
        urls   = _extract_result_urls(record)
        return VideoOutput(
            video_url = urls[0] if urls else "",
            model     = record.get("model", resolved),
            record_id = record.get("uuid"),
            metadata  = record,
        )

    async def swap_face_in_video(
        self,
        video_url: str,
        face_image_url: str,
    ) -> VideoOutput:
        """
        Swap a face in a video (``VIDEO_FACE_SWAPPER``).

        Parameters
        ----------
        video_url:
            URL or asset key of the source video.
        face_image_url:
            URL or asset key of the face image to transplant.
        """
        data   = await self._feature_post({
            "type":         "VIDEO_FACE_SWAPPER",
            "model":        "Qubico/video-toolkit",
            "promptObject": {
                "videoList":       [video_url],
                "faceImageList":   [face_image_url],
            },
        })
        record = data.get("aiRecord", {})
        urls   = _extract_result_urls(record)
        return VideoOutput(
            video_url = urls[0] if urls else "",
            model     = "Qubico/video-toolkit",
            record_id = record.get("uuid"),
            metadata  = record,
        )

    # ══════════════════════════════════════════════════════════════════════
    # AUDIO
    # ══════════════════════════════════════════════════════════════════════

    async def text_to_speech(
        self,
        text: str,
        model: str | TTSModel | None = None,
        *,
        voice: str | TTSVoice = TTSVoice.ALLOY,
        speed: float = 1.0,
    ) -> AudioOutput:
        """
        Convert *text* to speech (``TEXT_TO_SPEECH``).

        Parameters
        ----------
        text:
            Text to synthesize.
        model:
            Defaults to ``tts-1`` (OpenAI).
        voice:
            Voice name.  Use a :class:`~1minai.constants.TTSVoice` member
            or a raw string for ElevenLabs / Google voices.
        speed:
            Playback speed (0.25–4.0).

        Example::

            audio = await client.text_to_speech(
                "Cześć! Witaj w 1min.AI.",
                voice=TTSVoice.NOVA,
            )
            await audio.save(".")
        """
        resolved_model = _resolve(model, DEFAULT_TTS_MODEL)
        resolved_voice = voice.value if isinstance(voice, TTSVoice) else str(voice)
        data   = await self._feature_post({
            "type":         "TEXT_TO_SPEECH",
            "model":        resolved_model,
            "promptObject": {
                "prompt": text,
                "voice":  resolved_voice,
                "speed":  speed,
            },
        })
        record = data.get("aiRecord", {})
        urls   = _extract_result_urls(record)
        return AudioOutput(
            audio_url = urls[0] if urls else "",
            model     = record.get("model", resolved_model),
            record_id = record.get("uuid"),
            metadata  = record,
        )

    async def speech_to_text(
        self,
        audio_url: str,
        model: str | STTModel | None = None,
        *,
        language: str | None = None,
    ) -> TranscriptionOutput:
        """
        Transcribe audio to text (``SPEECH_TO_TEXT``).

        Parameters
        ----------
        audio_url:
            URL or asset key of the audio file.
        model:
            Defaults to ``whisper-1``.
        language:
            BCP-47 code, e.g. ``"en"``, ``"pl"``.  Auto-detected if omitted.
        """
        resolved   = _resolve(model, DEFAULT_STT_MODEL)
        prompt_obj: dict = {"audioList": [audio_url]}
        if language:
            prompt_obj["language"] = language
        data   = await self._feature_post({
            "type":         "SPEECH_TO_TEXT",
            "model":        resolved,
            "promptObject": prompt_obj,
        })
        record = data.get("aiRecord", {})
        return TranscriptionOutput(
            text      = _text_from_record(record),
            model     = record.get("model", resolved),
            record_id = record.get("uuid"),
            metadata  = record,
        )

    async def translate_audio(
        self,
        audio_url: str,
        target_language: str,
        model: str | STTModel | None = None,
    ) -> TranscriptionOutput:
        """
        Transcribe and translate audio into *target_language*
        (``AUDIO_TRANSLATOR``).
        """
        resolved = _resolve(model, "whisper-1")
        data     = await self._feature_post({
            "type":         "AUDIO_TRANSLATOR",
            "model":        resolved,
            "promptObject": {
                "audioList": [audio_url],
                "language":  target_language,
            },
        })
        record = data.get("aiRecord", {})
        return TranscriptionOutput(
            text      = _text_from_record(record),
            model     = record.get("model", resolved),
            record_id = record.get("uuid"),
            metadata  = record,
        )

    async def generate_captions(
        self,
        audio_or_video_url: str,
        model: str | STTModel | None = None,
        *,
        language: str | None = None,
    ) -> TranscriptionOutput:
        """
        Generate captions / subtitles for audio or video
        (``CAPTIONS_GENERATOR``).
        """
        resolved   = _resolve(model, "whisper-1")
        prompt_obj: dict = {"audioList": [audio_or_video_url]}
        if language:
            prompt_obj["language"] = language
        data   = await self._feature_post({
            "type":         "CAPTIONS_GENERATOR",
            "model":        resolved,
            "promptObject": prompt_obj,
        })
        record = data.get("aiRecord", {})
        return TranscriptionOutput(
            text      = _text_from_record(record),
            model     = record.get("model", resolved),
            record_id = record.get("uuid"),
            metadata  = record,
        )

    async def isolate_voice(self, audio_url: str) -> AudioOutput:
        """
        Isolate the voice track from background noise/music
        (``VOICE_ISOLATOR``).
        """
        data   = await self._feature_post({
            "type":         "VOICE_ISOLATOR",
            "model":        "elevenlabs-voice-isolator",
            "promptObject": {"audioList": [audio_url]},
        })
        record = data.get("aiRecord", {})
        urls   = _extract_result_urls(record)
        return AudioOutput(
            audio_url = urls[0] if urls else "",
            model     = "elevenlabs-voice-isolator",
            record_id = record.get("uuid"),
            metadata  = record,
        )

    async def change_voice(
        self,
        audio_url: str,
        target_voice: str,
    ) -> AudioOutput:
        """
        Transform the speaker's voice in an audio clip (``VOICE_CHANGER``).

        Parameters
        ----------
        audio_url:
            URL or asset key of the source audio.
        target_voice:
            Target voice ID (ElevenLabs voice name or ID).
        """
        data   = await self._feature_post({
            "type":         "VOICE_CHANGER",
            "model":        "elevenlabs-voice-changer",
            "promptObject": {
                "audioList":   [audio_url],
                "targetVoice": target_voice,
            },
        })
        record = data.get("aiRecord", {})
        urls   = _extract_result_urls(record)
        return AudioOutput(
            audio_url = urls[0] if urls else "",
            model     = "elevenlabs-voice-changer",
            record_id = record.get("uuid"),
            metadata  = record,
        )

    async def clone_voice(
        self,
        audio_url: str,
        name: str,
        description: str = "",
    ) -> AudioOutput:
        """
        Clone a voice from a reference audio clip (``VOICE_CLONING``).

        Parameters
        ----------
        audio_url:
            URL or asset key of the reference recording.
        name:
            Display name for the cloned voice.
        description:
            Optional description.
        """
        data   = await self._feature_post({
            "type":         "VOICE_CLONING",
            "model":        "elevenlabs-voice-cloning",
            "promptObject": {
                "audioList":   [audio_url],
                "name":        name,
                "description": description,
            },
        })
        record = data.get("aiRecord", {})
        urls   = _extract_result_urls(record)
        return AudioOutput(
            audio_url = urls[0] if urls else "",
            model     = "elevenlabs-voice-cloning",
            record_id = record.get("uuid"),
            metadata  = record,
        )

    async def design_voice(
        self,
        description: str,
        sample_text: str = "Hello, this is a voice design sample.",
    ) -> AudioOutput:
        """
        Design a new synthetic voice from a text description
        (``VOICE_DESIGN``).

        Parameters
        ----------
        description:
            Natural-language description of the voice (gender, age, accent…).
        sample_text:
            Text to synthesize with the designed voice for preview.
        """
        data   = await self._feature_post({
            "type":         "VOICE_DESIGN",
            "model":        "elevenlabs-voice-design",
            "promptObject": {
                "prompt":     description,
                "sampleText": sample_text,
            },
        })
        record = data.get("aiRecord", {})
        urls   = _extract_result_urls(record)
        return AudioOutput(
            audio_url = urls[0] if urls else "",
            model     = "elevenlabs-voice-design",
            record_id = record.get("uuid"),
            metadata  = record,
        )

    async def text_to_sound(
        self,
        prompt: str,
        duration: float = 5.0,
    ) -> AudioOutput:
        """
        Generate a sound effect from a text description (``TEXT_TO_SOUND``).

        Parameters
        ----------
        prompt:
            Description of the desired sound effect.
        duration:
            Length in seconds.
        """
        data   = await self._feature_post({
            "type":         "TEXT_TO_SOUND",
            "model":        "elevenlabs-text-to-sound",
            "promptObject": {"prompt": prompt, "duration": duration},
        })
        record = data.get("aiRecord", {})
        urls   = _extract_result_urls(record)
        return AudioOutput(
            audio_url = urls[0] if urls else "",
            model     = "elevenlabs-text-to-sound",
            record_id = record.get("uuid"),
            metadata  = record,
        )

    # ══════════════════════════════════════════════════════════════════════
    # MUSIC
    # ══════════════════════════════════════════════════════════════════════

    async def generate_music(
        self,
        prompt: str,
        model: str | MusicModel | None = None,
        *,
        duration: float | None = None,
        instrumental: bool = False,
    ) -> MusicOutput:
        """
        Generate music from a text prompt (``MUSIC_GENERATOR``).

        Parameters
        ----------
        prompt:
            Musical description (genre, mood, instruments, tempo…).
        model:
            Defaults to ``lyria-002`` (Google Lyria 2).
        duration:
            Target duration in seconds (model-dependent support).
        instrumental:
            If ``True``, suppress vocals where the model supports it.

        Example::

            track = await client.generate_music(
                "Upbeat lo-fi hip hop, calm and focused, 90 BPM",
            )
            await track.save(".")
        """
        resolved   = _resolve(model, DEFAULT_MUSIC_MODEL)
        prompt_obj: dict = {"prompt": prompt, "instrumental": instrumental}
        if duration is not None:
            prompt_obj["duration"] = duration
        data   = await self._feature_post({
            "type":         "MUSIC_GENERATOR",
            "model":        resolved,
            "promptObject": prompt_obj,
        })
        record = data.get("aiRecord", {})
        urls   = _extract_result_urls(record)
        return MusicOutput(
            audio_url = urls[0] if urls else "",
            model     = record.get("model", resolved),
            record_id = record.get("uuid"),
            metadata  = record,
        )

    # ══════════════════════════════════════════════════════════════════════
    # ASSETS
    # ══════════════════════════════════════════════════════════════════════

    async def upload_asset(
        self,
        file_path: str | Path | None = None,
        *,
        data: bytes | None = None,
        filename: str = "file",
        mime_type: str | None = None,
        asset_type: str | AssetType = AssetType.DOCUMENT,
    ) -> AssetRecord:
        """
        Upload a file to the team Asset API.

        Use the returned ``asset_key`` to reference images in image-API calls,
        and ``file_id`` in ``files=[]`` chat attachments.

        Parameters
        ----------
        file_path:
            Local path to the file.
        data:
            Raw bytes (alternative to *file_path*).
        filename:
            File name to send when uploading raw bytes.
        mime_type:
            MIME type; auto-detected from *file_path* if omitted.
        asset_type:
            Asset category.  Defaults to ``DOCUMENT``.

        Returns
        -------
        AssetRecord

        Example::

            asset = await client.upload_asset("report.pdf")
            reply = await client.chat(
                "Summarise this document.", files=[asset.file_id]
            )
        """
        resolved_type = (
            asset_type.value if isinstance(asset_type, AssetType) else str(asset_type)
        )

        if data is None:
            path      = Path(file_path)  # type: ignore[arg-type]
            mime_type = mime_type or mimetypes.guess_type(str(path))[0] or "application/octet-stream"
            data      = path.read_bytes()
            filename  = path.name

        mime_type = mime_type or "application/octet-stream"
        await self._get_team_id()
        session = await self._get_session()
        url     = self._team_url(_ASSET_URL)

        form = aiohttp.FormData()
        form.add_field("asset", data, filename=filename, content_type=mime_type)
        form.add_field("type", resolved_type)

        async with session.post(
            url,
            data=form,
            timeout=aiohttp.ClientTimeout(total=120),
            proxy=self._proxy,
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                raise AssetUploadError(
                    f"Upload failed ({resp.status}): {body[:300]}"
                )
            result = await resp.json(content_type=None)

        # Real response shape: {"asset": {..., "key": "..."}, "fileContent": {"uuid": "..."}}
        asset_part   = result.get("asset", result)
        content_part = result.get("fileContent", {})

        asset_key = asset_part.get("key", "")
        file_id   = content_part.get("uuid", asset_part.get("uuid", asset_part.get("id", "")))

        logger.info("Uploaded %s → key=%s  file_id=%s", filename, asset_key[:20], file_id[:8])
        return AssetRecord(
            asset_key  = asset_key,
            file_id    = file_id,
            asset_type = resolved_type,
            metadata   = result,
        )

    async def upload_asset_from_url(
        self,
        url: str,
        asset_type: str | AssetType = AssetType.IMAGE,
    ) -> AssetRecord:
        """
        Register a publicly accessible URL as a team asset.

        Parameters
        ----------
        url:
            Public URL to register.
        asset_type:
            Asset category.
        """
        resolved_type = (
            asset_type.value if isinstance(asset_type, AssetType) else str(asset_type)
        )
        await self._get_team_id()
        result = await self._post(
            self._team_url(_ASSET_URL),
            {"fileUrl": url, "type": resolved_type},
        )
        asset_part   = result.get("asset", result)
        content_part = result.get("fileContent", {})
        return AssetRecord(
            asset_key  = asset_part.get("key", ""),
            file_id    = content_part.get("uuid", asset_part.get("uuid", "")),
            asset_type = resolved_type,
            metadata   = result,
        )

    # ══════════════════════════════════════════════════════════════════════
    # TAGS
    # ══════════════════════════════════════════════════════════════════════

    async def list_tags(self) -> list[dict]:
        """Return all conversation tags for the current team."""
        await self._get_team_id()
        data = await self._get_json(self._team_url(_TAGS_URL))
        return data.get("tagList", [])  # type: ignore[union-attr]

    # ══════════════════════════════════════════════════════════════════════
    # EXPLORE / SOCIAL FEED
    # ══════════════════════════════════════════════════════════════════════

    async def get_explore_posts(self) -> list[dict]:
        """
        Return the public explore feed of AI-generated content.

        Each dict contains the post's attachments (images, prompts, models
        used), author info, hashtags, and visibility.
        """
        data = await self._get_json(_EXPLORE_URL)
        return data.get("data", data)  # type: ignore[union-attr]
