"""
LLM-backed symbol preprocessing functionality.

This module provides LLM-assisted preprocessing of text to mark potential symbols
with backticks for further processing by rules-based matching.
"""

from typing import List

from .llm_handler import get_llm_handler


class LLMMapError(Exception):
    """Raised when LLM symbol mapping fails."""

    pass


def llm_preprocess_text(transcript: str, candidate_symbols: List[str]) -> str:
    """
    Use LLM to preprocess text by marking potential symbols with backticks.

    This is optimized for hybrid mode where LLM runs first with filtered symbols,
    then rules process only the marked content.

    Args:
        transcript: Original transcript text
        candidate_symbols: Filtered list of relevant project symbols

    Returns:
        Text with potential symbols marked with backticks

    Raises:
        LLMMapError: If LLM call fails
    """
    if not transcript.strip() or not candidate_symbols:
        return transcript

    # Create specialized prompts for preprocessing
    system_prompt = _create_preprocessing_system_prompt(candidate_symbols)
    user_prompt = _create_preprocessing_user_prompt(transcript)

    # Use centralized LLM handler for the request
    llm_handler = get_llm_handler(".")
    return llm_handler.make_reconciliation_request(transcript, candidate_symbols)


# These functions are now handled by LLM Handler
# Keeping them for backward compatibility if needed elsewhere
def _create_preprocessing_system_prompt(candidate_symbols: List[str]) -> str:
    """Create system prompt for LLM text preprocessing."""
    from .llm_handler import get_llm_handler

    handler = get_llm_handler()
    return handler._create_reconciliation_system_prompt(candidate_symbols)


def _create_preprocessing_user_prompt(transcript: str) -> str:
    """Create user prompt for LLM text preprocessing."""
    from .llm_handler import get_llm_handler

    handler = get_llm_handler()
    return handler._create_reconciliation_user_prompt(transcript)
