from __future__ import annotations

from typing import TYPE_CHECKING
from collections.abc import AsyncIterator

from .types import ChatOutput

if TYPE_CHECKING:
    from .client import OneMinAIClient


class ChatSession:
    """
    An ongoing multi-turn conversation with a 1min.AI model.

    Do not instantiate directly — use :meth:`OneMinAIClient.start_chat`.

    The session is backed by a **server-side conversation** (persistent,
    history stored by 1min.AI).  A new server-side thread is created
    automatically on the first :meth:`send_message` call if no
    ``conversation_id`` was supplied.

    Example::

        chat = client.start_chat(model="gpt-4.1-nano")
        r1 = await chat.send_message("My name is Alice.")
        r2 = await chat.send_message("What's my name?")
        print(r2.text)   # "Your name is Alice."
    """

    def __init__(
        self,
        client: "OneMinAIClient",
        model: str,
        conversation_id: str | None = None,
        web_search: bool = False,
        num_of_site: int = 2,
        max_word: int = 1000,
        is_mixed: bool = False,
        history_message_limit: int = 8,
        with_memories: bool = False,
        brand_voice_id: str | None = None,
    ) -> None:
        self._client                = client
        self._model                 = model
        self._conversation_id       = conversation_id  # None until first send
        self._web_search            = web_search
        self._num_of_site           = num_of_site
        self._max_word              = max_word
        self._is_mixed              = is_mixed
        self._history_message_limit = history_message_limit
        self._with_memories         = with_memories
        self._brand_voice_id        = brand_voice_id
        self._last_output: ChatOutput | None = None
        self._turn_count: int = 0

    # ── properties ────────────────────────────────────────────────────────

    @property
    def conversation_id(self) -> str | None:
        """
        Server-side conversation UUID.

        ``None`` until the first message is sent (the server assigns the UUID
        on creation).
        """
        return self._conversation_id

    @property
    def model(self) -> str:
        """Model ID used by this session."""
        return self._model

    @property
    def last_output(self) -> ChatOutput | None:
        """The most recent :class:`~oneminai_webapi.types.ChatOutput` from this session."""
        return self._last_output

    @property
    def turn_count(self) -> int:
        """Number of messages sent in this session so far."""
        return self._turn_count

    # ── internal ──────────────────────────────────────────────────────────

    def _sync_conv_id(self, output: ChatOutput) -> None:
        """Update the cached conversation ID from a server response."""
        if output.conversation_id and output.conversation_id != self._conversation_id:
            self._conversation_id = output.conversation_id

    # ── send ──────────────────────────────────────────────────────────────

    async def send_message(
        self,
        prompt: str,
        *,
        images: list[str] | None = None,
        files: list[str] | None = None,
        model: str | None = None,
        web_search: bool | None = None,
    ) -> ChatOutput:
        """
        Send *prompt* and wait for the complete response.

        Parameters
        ----------
        prompt:
            The user message.
        images:
            Image URLs or asset keys to attach.
        files:
            File IDs from the Asset API to attach (PDFs, documents).
        model:
            Override the session-level model for this turn only.
        web_search:
            Override the session-level web-search flag for this turn only.

        Returns
        -------
        ChatOutput
        """
        output = await self._client.chat(
            prompt,
            stream                = False,
            model                 = model or self._model,
            conversation_id       = self._conversation_id,
            images                = images,
            files                 = files,
            web_search            = web_search if web_search is not None else self._web_search,
            num_of_site           = self._num_of_site,
            max_word              = self._max_word,
            is_mixed              = self._is_mixed,
            history_message_limit = self._history_message_limit,
            with_memories         = self._with_memories,
            brand_voice_id        = self._brand_voice_id,
        )
        self._sync_conv_id(output)
        self._last_output  = output
        self._turn_count  += 1
        return output

    async def send_message_stream(
        self,
        prompt: str,
        *,
        images: list[str] | None = None,
        files: list[str] | None = None,
        model: str | None = None,
        web_search: bool | None = None,
    ) -> AsyncIterator[ChatOutput]:
        """
        Stream the response to *prompt*, yielding incremental
        :class:`~oneminai_webapi.types.ChatOutput` chunks.

        ``chunk.text_delta`` contains only the new text since the previous
        chunk.  ``chunk.text`` contains the full accumulated text so far.

        Example::

            async for chunk in chat.send_message_stream("Write me an essay"):
                print(chunk.text_delta, end="", flush=True)
            print()
        """
        async for chunk in await self._client.chat(
            prompt,
            stream                = True,
            model                 = model or self._model,
            conversation_id       = self._conversation_id,
            images                = images,
            files                 = files,
            web_search            = web_search if web_search is not None else self._web_search,
            num_of_site           = self._num_of_site,
            max_word              = self._max_word,
            is_mixed              = self._is_mixed,
            history_message_limit = self._history_message_limit,
            with_memories         = self._with_memories,
            brand_voice_id        = self._brand_voice_id,
        ):
            self._sync_conv_id(chunk)
            self._last_output = chunk
            yield chunk
        self._turn_count += 1

    # ── history ───────────────────────────────────────────────────────────

    async def get_history(self) -> list:
        """
        Fetch the full message history from the server.

        Returns
        -------
        list[MessageRecord]
            Empty list if the conversation has not been created yet.
        """
        if not self._conversation_id:
            return []
        return await self._client.get_conversation_messages(self._conversation_id)

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def delete(self) -> None:
        """
        Delete this conversation from the server.

        No-op if the session has not sent any messages yet (no server-side
        conversation exists).
        """
        if self._conversation_id:
            await self._client.delete_conversation(self._conversation_id)
            self._conversation_id = None
            self._turn_count      = 0

    # ── repr ──────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        cid = self._conversation_id[:8] + "…" if self._conversation_id else "pending"
        return (
            f"ChatSession(model={self._model!r}, "
            f"conversation_id={cid!r}, "
            f"turns={self._turn_count})"
        )