"""
LLM-backed symbol preprocessing functionality.

This module provides LLM-assisted preprocessing of text to mark potential symbols
with backticks for further processing by rules-based matching.
"""

from typing import List

from .config import config, get_client


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

    # Debug logging - import here to avoid circular imports
    from .debug_log import get_debug_logger

    debug_logger = get_debug_logger(".")

    try:
        from .progress import reporter

        reporter.sub_step_with_progress("LLM preprocessing", "analyzing text for potential symbols", 1, 2)

        # Debug logging removed - only essential logging kept
        full_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"

        client = get_client()

        response = client.chat.completions.create(
            model=config.distil_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            temperature=0.1,  # Low temperature for consistent marking
            max_tokens=2000,  # Allow for longer responses with marked text
        )

        content = response.choices[0].message.content
        if not content:
            # Return original text if empty response
            return transcript

        # Debug logging removed - content processed directly

        reporter.sub_step_with_progress("LLM preprocessing", "extracting marked text", 2, 2)
        return content.strip()

    except Exception:
        # If LLM preprocessing fails, return original text
        return transcript


def _create_preprocessing_system_prompt(candidate_symbols: List[str]) -> str:
    """Create system prompt for LLM text preprocessing."""
    symbols_text = ", ".join(f"`{symbol}`" for symbol in candidate_symbols)

    return f"""You are an expert at identifying code symbols in natural language text across multiple languages.

Your task is to read a transcript and mark any phrases that likely refer to these project symbols with backticks.

AVAILABLE SYMBOLS: {symbols_text}

MULTILINGUAL PROCESSING INSTRUCTIONS:
1. The transcript may be in ANY language (English, Russian, Chinese, Spanish, etc.)
2. Pay special attention to ANGLICISMS - English technical terms used in other languages
3. Look for phonetic adaptations of English terms (e.g., "процессор" → "processor")
4. Consider direct translations and conceptual matches
5. Technical terms are often borrowed across languages with slight modifications

INSTRUCTIONS:
1. Read the transcript carefully, regardless of language
2. Identify phrases that might refer to the symbols above
3. Mark ONLY those phrases with backticks (e.g., `symbol_name`)
4. Consider ALL variations: snake_case ↔ CamelCase, plurals, verb forms, spaces, translations
5. Preserve all other text exactly as written
6. If unsure, don't mark it

MULTILINGUAL EXAMPLES:
English: "update delete task" → "update `delete_task`"
Russian: "обнови удаление задачи" → "обнови `delete_task`"
Russian (with anglicisms): "фикс спич процессор" → "фикс `SpeechProcessor`"
Spanish: "actualizar manejador de login" → "actualizar `login_handler`"
Chinese: "修复配置错误" → "修复`ConfigError`"

TECHNICAL TERM MATCHING:
- processor/процессор/处理器 → *Processor
- handler/обработчик/处理程序 → *Handler
- error/ошибка/错误 → *Error
- config/конфиг/配置 → *Config
- test/тест/测试 → *Test*

Return the complete text with appropriate symbols marked with backticks."""


def _create_preprocessing_user_prompt(transcript: str) -> str:
    """Create user prompt for LLM text preprocessing."""
    return f"""Mark potential code symbols in this transcript with backticks:

{transcript}

Return the complete text with symbols marked."""
