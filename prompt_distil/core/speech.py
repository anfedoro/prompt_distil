"""
Speech-to-text functionality using OpenAI Whisper.

This module provides a wrapper around OpenAI's Whisper ASR API
for converting audio files to text transcripts.
"""

import re
from pathlib import Path

from .config import config, get_client
from .types import Transcript


class SpeechError(Exception):
    """Raised when speech processing fails."""

    pass


# Domain whitelist for code identifiers
CODE_DOMAIN_WHITELIST = {
    "delete_task",
    "login_handler",
    "pytest",
    "pydantic",
    "FastAPI",
    "EmailStr",
    "user_model",
    "auth_flow",
    "test_case",
    "api_key",
    "db_session",
    "config_file",
    "error_handler",
    "middleware",
    "async_task",
    "data_model",
    "response_type",
}


def protect_code_identifiers(text: str) -> str:
    """
    Protect code identifiers by wrapping them with backticks.

    Applies heuristics to identify code-like tokens:
    - Contains underscore OR
    - Is in the domain whitelist

    Args:
        text: Input text to process

    Returns:
        Text with protected code identifiers wrapped in backticks
    """
    # Pattern to match potential identifiers: [A-Za-z_][A-Za-z0-9_]*
    identifier_pattern = r"\b[A-Za-z_][A-Za-z0-9_]*\b"

    def should_protect(match):
        """Determine if an identifier should be protected."""
        identifier = match.group(0)

        # Skip if already backticked
        start_pos = match.start()
        end_pos = match.end()

        if start_pos > 0 and text[start_pos - 1] == "`":
            return False
        if end_pos < len(text) and text[end_pos] == "`":
            return False

        # Protect if contains underscore or in whitelist (case insensitive)
        return "_" in identifier or identifier in CODE_DOMAIN_WHITELIST or identifier.lower() in CODE_DOMAIN_WHITELIST

    def replace_identifier(match):
        """Replace identifier with backticked version if should be protected."""
        if should_protect(match):
            return f"`{match.group(0)}`"
        return match.group(0)

    return re.sub(identifier_pattern, replace_identifier, text)


class SpeechProcessor:
    """
    Handles speech-to-text conversion using OpenAI Whisper.

    Uses automatic language detection and transcription only (no translation).
    Final language processing is handled by the distiller LLM.
    """

    def __init__(self):
        self.client = get_client()
        self.model = config.asr_model

    def transcribe_audio(self, path: str) -> Transcript:
        """
        Transcribe audio file to text using Whisper with auto language detection.

        Uses transcription-only policy: no translation at ASR level.
        Final language handling is performed by the distiller LLM.

        Args:
            path: Path to the audio file

        Returns:
            Transcript object with text and detected language

        Raises:
            SpeechError: If transcription fails
            FileNotFoundError: If audio file doesn't exist
        """
        audio_path = Path(path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        if not audio_path.is_file():
            raise SpeechError(f"Path is not a file: {path}")

        # Check file size (Whisper has a 25MB limit)
        file_size = audio_path.stat().st_size
        max_size = 25 * 1024 * 1024  # 25MB in bytes

        if file_size > max_size:
            raise SpeechError(f"Audio file too large: {file_size / 1024 / 1024:.1f}MB (max: 25MB)")

        try:
            with open(audio_path, "rb") as audio_file:
                # Always use transcription endpoint - never translation
                # Let Whisper auto-detect the language
                response = self.client.audio.transcriptions.create(model=self.model, file=audio_file, response_format="verbose_json")

                # Extract text and detected language
                if hasattr(response, "text"):
                    text = response.text.strip()
                    # Get detected language from Whisper response
                    lang_detected = getattr(response, "language", "auto")
                else:
                    text = str(response).strip()
                    lang_detected = "auto"

                return Transcript(text=text, lang_hint=lang_detected)

        except Exception as e:
            raise SpeechError(f"Failed to transcribe audio: {str(e)}")

    def validate_audio_format(self, path: str) -> bool:
        """
        Validate if the audio file format is supported.

        Args:
            path: Path to the audio file

        Returns:
            True if format is supported, False otherwise
        """
        supported_extensions = {".mp3", ".mp4", ".mpeg", ".mpga", ".m4a", ".wav", ".webm"}

        file_path = Path(path)
        extension = file_path.suffix.lower()

        return extension in supported_extensions

    def get_audio_info(self, path: str) -> dict:
        """
        Get basic information about the audio file.

        Args:
            path: Path to the audio file

        Returns:
            Dictionary with file information
        """
        audio_path = Path(path)

        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {path}")

        stat = audio_path.stat()

        return {
            "path": str(audio_path.absolute()),
            "name": audio_path.name,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / 1024 / 1024, 2),
            "extension": audio_path.suffix.lower(),
            "supported": self.validate_audio_format(path),
        }


# Convenience function for simple use cases
def transcribe_audio(path: str) -> Transcript:
    """
    Convenience function to transcribe audio file.

    Args:
        path: Path to the audio file

    Returns:
        Transcript object with protected code identifiers

    Raises:
        SpeechError: If transcription fails
        FileNotFoundError: If audio file doesn't exist
    """
    processor = SpeechProcessor()
    transcript = processor.transcribe_audio(path)

    # Always protect code identifiers in transcripts
    protected_text = protect_code_identifiers(transcript.text)
    transcript = Transcript(text=protected_text, lang_hint=transcript.lang_hint)

    return transcript
