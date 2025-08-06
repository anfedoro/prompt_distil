"""
Configuration management for the Prompt Distiller.

This module handles environment variables, API keys, and model configurations
using python-dotenv for loading .env files.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from openai import OpenAI


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


@lru_cache(maxsize=1)
def load_config() -> None:
    """Load environment variables from .env file if it exists."""
    load_dotenv()


class Config:
    """Configuration settings for the Prompt Distiller."""

    def __init__(self):
        load_config()

    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from environment."""
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ConfigError("OPENAI_API_KEY not found in environment. Please set it in your environment or .env file.")
        return key

    @property
    def distil_model(self) -> str:
        """Get the model name for distillation (default: gpt-4o)."""
        return os.getenv("DISTIL_MODEL", "gpt-4.1-mini")

    @property
    def asr_model(self) -> str:
        """Get the model name for ASR (default: whisper-1)."""
        return os.getenv("ASR_MODEL", "whisper-1")

    @property
    def openai_timeout(self) -> int:
        """Get OpenAI API timeout in seconds (default: 60)."""
        return int(os.getenv("OPENAI_TIMEOUT", "60"))

    @property
    def max_retries(self) -> int:
        """Get maximum number of retries for API calls (default: 3)."""
        return int(os.getenv("MAX_RETRIES", "3"))


# Global config instance
config = Config()


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """
    Get configured OpenAI client with timeout and retry settings.

    Returns:
        OpenAI client instance

    Raises:
        ConfigError: If API key is not configured
    """
    try:
        return OpenAI(
            api_key=config.openai_api_key,
            timeout=config.openai_timeout,
            max_retries=config.max_retries,
        )
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(f"Failed to create OpenAI client: {e}")


def validate_config() -> None:
    """
    Validate that all required configuration is present.

    Raises:
        ConfigError: If required configuration is missing
    """
    # This will raise ConfigError if API key is missing
    _ = config.openai_api_key

    # Test client creation
    _ = get_client()
