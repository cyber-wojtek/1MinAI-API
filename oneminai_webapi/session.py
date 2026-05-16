from __future__ import annotations

from typing import TYPE_CHECKING, AsyncIterator

from .types import ChatOutput

if TYPE_CHECKING:
    from .client import OneMinAIClient


class ChatSession:
    """
    An ongoing multi-turn conversation with a 1min.AI model.

    Do not instantiate directly — use :meth:`OneMinAIClient.start_chat`.

    The session can be backed by a **server-side conversation** (persistent,
    history stored by 1min.AI) or run **statelessly** (history managed by the
    ``historyMessageLimit`` context window without a persisted thread).

    Example::

        chat = await client.start_chat(model="gpt-4.1-nano")
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
        self._conversation_id       = conversation_id
        self._web_search            = web_search
        self._num_of_site           = num_of_site
        self._max_word              = max_word
        self._is_mixed              = is_mixed
        self._history_message_limit = history_message_limit
        self._with_memories         = with_memories
        self._brand_voice_id        = brand_voice_id
        self._last_output: ChatOutput | None = None

    # ── properties ────────────────────────────────────────────────────────

    @property
    def conversation_id(self) -> str | None:
        """Server-side conversation UUID, or ``None`` for stateless sessions."""
        return self._conversation_id

    @property
    def model(self) -> str:
        """Model ID used by this session."""
        return self._model

    @property
    def last_output(self) -> ChatOutput | None:
        """The most recent :class:`~1minai.types.ChatOutput` from this session."""
        return self._last_output

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
            Override the session-level web-search flag for this turn.

        Returns
        -------
        ChatOutput
        """
        output = await self._client.chat(
            prompt,
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
            stream                = False,
        )
        # Keep the conversation_id in sync if the server assigned one.
        if output.conversation_id and not self._conversation_id:
            self._conversation_id = output.conversation_id
        self._last_output = output
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
        Stream *prompt* response, yielding incremental :class:`ChatOutput` chunks.

        The ``text_delta`` attribute of each chunk contains only the new text
        since the previous chunk.

        Example::

            async for chunk in chat.send_message_stream("Write me an essay"):
                print(chunk.text_delta, end="", flush=True)
        """
        async for chunk in await self._client.chat(
            prompt,
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
            stream                = True,
        ):
            if chunk.conversation_id and not self._conversation_id:
                self._conversation_id = chunk.conversation_id
            self._last_output = chunk
            yield chunk

    # ── lifecycle ─────────────────────────────────────────────────────────

    async def delete(self) -> None:
        """
        Delete this conversation from the server (no-op if stateless).
        """
        if self._conversation_id:
            await self._client.delete_conversation(self._conversation_id)
            self._conversation_id = None