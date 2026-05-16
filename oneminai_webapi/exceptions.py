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