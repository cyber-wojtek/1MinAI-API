"""
Custom exceptions for 1minai.
"""
from __future__ import annotations


class OneMinAIError(Exception):
    """Base exception for all 1minai errors."""


class AuthenticationError(OneMinAIError):
    """Raised when the API key or JWT token is invalid or missing."""


class APIError(OneMinAIError):
    """Raised when the 1min.AI API returns an unexpected HTTP status code."""

    def __init__(self, message: str, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class TimeoutError(OneMinAIError):
    """Raised when a request to the 1min.AI API times out."""


class AssetUploadError(OneMinAIError):
    """Raised when an asset upload fails."""


class RateLimitError(OneMinAIError):
    """Raised when the API rate limit is exceeded (HTTP 429)."""

    def __init__(self, message: str, retry_after_s: int | None = None) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


class ConversationError(OneMinAIError):
    """Raised when a conversation operation fails."""


class ValidationError(OneMinAIError):
    """Raised when request parameters fail server-side validation (HTTP 422)."""

    def __init__(self, message: str, details: list[dict] | None = None) -> None:
        super().__init__(message)
        self.details: list[dict] = details or []


class OAuthError(OneMinAIError):
    """
    Raised when the OAuth exchange with 1min.AI fails.

    Covers POST /auth/oauth non-2xx responses and responses that are missing
    ``user.token``.
    """


class CloudflareError(OneMinAIError):
    """
    Raised when a Cloudflare challenge or block is encountered.

    Attributes
    ----------
    status_code:
        HTTP status code returned (typically 403, 429, or 503).
    challenge_type:
        The type of Cloudflare challenge detected, e.g.
        ``"js_challenge"``, ``"managed_challenge"``, ``"turnstile"``,
        ``"block"``, ``"rate_limit"``, or ``"unknown"``.
    ray_id:
        The ``CF-RAY`` header value, useful for reporting to Cloudflare.
        ``None`` if the header was absent.

    Resolution
    ----------
    Pass a fresh ``cf_clearance`` cookie (and matching ``user_agent``) to
    :class:`~oneminai_webapi.OneMinAIClient`::

        client = OneMinAIClient(
            api_key="…",
            cf_clearance="abc123xyz…",
            user_agent="Mozilla/5.0 …",
        )
    """

    def __init__(
        self,
        message: str,
        status_code: int | None = None,
        challenge_type: str = "unknown",
        ray_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code    = status_code
        self.challenge_type = challenge_type
        self.ray_id         = ray_id

    def __str__(self) -> str:
        parts = [super().__str__()]
        if self.challenge_type != "unknown":
            parts.append(f"challenge_type={self.challenge_type!r}")
        if self.ray_id:
            parts.append(f"ray_id={self.ray_id!r}")
        if self.status_code:
            parts.append(f"status_code={self.status_code}")
        return "  ".join(parts)

