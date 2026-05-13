"""
Security utilities for SemanticCite.

This module provides utilities for sanitizing error messages and handling
sensitive information securely.
"""

import re
import logging
from typing import Any, Dict, Optional


# Patterns to detect and redact sensitive information
SENSITIVE_PATTERNS = [
    # API Keys (various formats)
    (r'(api[_-]?key[\'"\s:=]+)([a-zA-Z0-9_\-]{20,})', r'\1[REDACTED]'),
    (r'(sk-[a-zA-Z0-9]{20,})', r'[REDACTED_API_KEY]'),  # OpenAI style
    (r'(AIza[a-zA-Z0-9_\-]{35})', r'[REDACTED_API_KEY]'),  # Google API keys

    # Bearer Tokens
    (r'(Bearer\s+)([a-zA-Z0-9_\-\.]{20,})', r'\1[REDACTED_TOKEN]'),
    (r'(Authorization[\'"\s:=]+)([^\s\'"]+)', r'\1[REDACTED]'),

    # Access Tokens
    (r'(access[_-]?token[\'"\s:=]+)([a-zA-Z0-9_\-\.]{20,})', r'\1[REDACTED]'),
    (r'(token[\'"\s:=]+)([a-zA-Z0-9_\-\.]{20,})', r'\1[REDACTED]'),

    # API Key Environment Variables (values)
    (r'(OPENAI_API_KEY[\'"\s:=]+)([^\s\'"]+)', r'\1[REDACTED]'),
    (r'(ANTHROPIC_API_KEY[\'"\s:=]+)([^\s\'"]+)', r'\1[REDACTED]'),
    (r'(GEMINI_API_KEY[\'"\s:=]+)([^\s\'"]+)', r'\1[REDACTED]'),
    (r'(NVIDIA_API_KEY[\'"\s:=]+)([^\s\'"]+)', r'\1[REDACTED]'),

    # URLs with embedded credentials
    (r'(https?://[^:]+:)([^@]+)(@)', r'\1[REDACTED]\3'),

    # Generic secrets patterns
    (r'(secret[\'"\s:=]+)([a-zA-Z0-9_\-]{10,})', r'\1[REDACTED]'),
    (r'(password[\'"\s:=]+)([^\s\'"]+)', r'\1[REDACTED]'),
]


def sanitize_error_message(error: Any) -> str:
    """
    Sanitize error messages by removing sensitive information like API keys.

    Args:
        error: Exception or error message to sanitize

    Returns:
        Sanitized error message safe for display/logging
    """
    # Convert error to string
    error_str = str(error)

    # Apply all sanitization patterns
    for pattern, replacement in SENSITIVE_PATTERNS:
        error_str = re.sub(pattern, replacement, error_str, flags=re.IGNORECASE)

    return error_str


def get_user_friendly_error(error: Any, context: str = "operation") -> str:
    """
    Generate a user-friendly error message without exposing sensitive details.

    Args:
        error: The original exception
        context: Context of the operation (e.g., "API call", "file processing")

    Returns:
        User-friendly error message
    """
    error_str = str(error).lower()

    # Map common error patterns to user-friendly messages
    # Check more specific patterns first, before more generic ones
    if 'rate limit' in error_str or 'quota' in error_str:
        return f"API rate limit exceeded. Please try again later."

    elif 'authentication' in error_str or 'unauthorized' in error_str or 'api key' in error_str:
        return f"Authentication failed. Please check your API key configuration."

    elif 'timeout' in error_str:
        return f"The {context} timed out. Please try again."

    elif 'network' in error_str or 'connection' in error_str:
        return f"Network error occurred. Please check your internet connection."

    elif 'not found' in error_str or '404' in error_str:
        return f"Resource not found. Please check the URL or file path."

    elif 'permission' in error_str or 'forbidden' in error_str:
        return f"Permission denied. Please check access permissions."

    elif 'invalid' in error_str:
        return f"Invalid input provided. Please check your data."

    else:
        return f"An error occurred during {context}. Please try again or contact support."


class SecureLogger:
    """
    Logger wrapper that automatically sanitizes sensitive information.
    """

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def _sanitize_log_data(self, msg: str, *args, **kwargs) -> tuple:
        """Sanitize message and arguments before logging."""
        sanitized_msg = sanitize_error_message(msg)
        sanitized_args = tuple(sanitize_error_message(str(arg)) for arg in args)
        sanitized_kwargs = {k: sanitize_error_message(str(v)) for k, v in kwargs.items()}
        return sanitized_msg, sanitized_args, sanitized_kwargs

    def debug(self, msg: str, *args, **kwargs):
        msg, args, kwargs = self._sanitize_log_data(msg, *args, **kwargs)
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg: str, *args, **kwargs):
        msg, args, kwargs = self._sanitize_log_data(msg, *args, **kwargs)
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg: str, *args, **kwargs):
        msg, args, kwargs = self._sanitize_log_data(msg, *args, **kwargs)
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg: str, *args, **kwargs):
        msg, args, kwargs = self._sanitize_log_data(msg, *args, **kwargs)
        self.logger.error(msg, *args, **kwargs)

    def critical(self, msg: str, *args, **kwargs):
        msg, args, kwargs = self._sanitize_log_data(msg, *args, **kwargs)
        self.logger.critical(msg, *args, **kwargs)


def handle_exception_safely(
    exception: Exception,
    context: str = "operation",
    log_full_error: bool = True,
    logger: Optional[logging.Logger] = None
) -> Dict[str, str]:
    """
    Handle an exception safely by separating user-facing and internal logging.

    Args:
        exception: The exception to handle
        context: Context of the operation
        log_full_error: Whether to log the full sanitized error internally
        logger: Logger instance to use (optional)

    Returns:
        Dictionary with 'user_message' and 'log_message' keys
    """
    # Generate user-friendly message (no sensitive data)
    user_message = get_user_friendly_error(exception, context)

    # Generate sanitized log message (sensitive data redacted)
    log_message = sanitize_error_message(exception)

    # Log internally if requested
    if log_full_error and logger:
        logger.error(f"Error in {context}: {log_message}")

    return {
        'user_message': user_message,
        'log_message': log_message,
        'error_type': type(exception).__name__
    }