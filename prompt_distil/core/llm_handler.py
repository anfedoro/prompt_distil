"""
Centralized LLM Handler for all LLM requests in the prompt_distil system.

This module provides a unified interface for making LLM requests with automatic
error handling, reasoning model fallback, and request-specific parameter adjustments.
"""

import json
import logging
import os
from typing import Any, Dict, List, Optional

from .config import config, get_client
from .debug_log import get_debug_logger

logger = logging.getLogger(__name__)


class ReasoningModelError(Exception):
    """Raised when reasoning model parameter adjustment fails."""

    pass


def get_env_reasoning_effort() -> str:
    """Get reasoning effort from environment variable with default fallback."""
    effort = os.getenv("REASONING_EFFORT", "low").lower()
    valid_efforts = {"minimal", "low", "medium", "high"}
    if effort not in valid_efforts:
        logger.warning(f"Invalid REASONING_EFFORT value: {effort}. Using 'low' as default.")
        return "low"
    return effort


def get_env_verbosity() -> str:
    """Get verbosity from environment variable with default fallback."""
    verbosity = os.getenv("VERBOSITY", "medium").lower()
    valid_verbosity = {"low", "medium", "high"}
    if verbosity not in valid_verbosity:
        logger.warning(f"Invalid VERBOSITY value: {verbosity}. Using 'medium' as default.")
        return "medium"
    return verbosity


def get_env_model_temperature() -> float:
    """Get model temperature from environment variable with default fallback."""
    try:
        temp = float(os.getenv("MODEL_TEMPERATURE", "0.2"))
        if not (0.0 <= temp <= 2.0):
            logger.warning(f"Invalid MODEL_TEMPERATURE value: {temp}. Using 0.2 as default.")
            return 0.2
        return temp
    except (ValueError, TypeError):
        logger.warning("Invalid MODEL_TEMPERATURE format. Using 0.2 as default.")
        return 0.2


def is_reasoning_model_error(exception: Exception) -> bool:
    """
    Check if the exception indicates a model type change to reasoning model.

    Detects error code 400 with parameters:
    - type: 'invalid request error'
    - code: 'unsupported value'
    - param: 'temperature'

    Args:
        exception: Exception from LLM API call

    Returns:
        True if this is a reasoning model error that needs parameter adjustment
    """
    try:
        # Check if it's an OpenAI API error with status code 400
        if hasattr(exception, "status_code") and exception.status_code == 400:
            # Try to get error details from different possible attributes
            error_data = None

            if hasattr(exception, "response") and hasattr(exception.response, "json"):
                try:
                    error_data = exception.response.json()
                except:
                    pass
            elif hasattr(exception, "body"):
                error_data = exception.body
            elif hasattr(exception, "error"):
                error_data = exception.error

            if error_data and isinstance(error_data, dict):
                error_info = error_data.get("error", {})

                # Check for the specific error pattern
                error_type = error_info.get("type", "").lower()
                error_code = error_info.get("code", "").lower()
                error_param = error_info.get("param", "").lower()

                # Match the specific patterns for reasoning model errors
                if (
                    error_type == "invalid_request_error"
                    and (error_code == "unsupported_value" or error_code == "unsupported_parameter")
                    and (error_param == "temperature" or error_param == "max_tokens")
                ):
                    return True

        return False

    except Exception as e:
        logger.debug(f"Error checking reasoning model error pattern: {e}")
        return False


def get_reasoning_model_parameters(request_type: str) -> Dict[str, Any]:
    """
    Get appropriate parameters for reasoning model requests using environment variables.

    Note: OpenAI reasoning models don't currently support custom reasoning_effort
    or verbosity parameters in the API. This function is prepared for future support.

    Args:
        request_type: Type of request - 'reconciliation' or 'distillation'

    Returns:
        Dictionary of parameters to use for reasoning model (currently empty)
    """
    # For now, reasoning models don't support custom parameters
    # This is kept for future OpenAI API updates
    return {}


def adjust_llm_params_for_reasoning_model(original_params: Dict[str, Any], request_type: str) -> Dict[str, Any]:
    """
    Adjust LLM request parameters for reasoning model compatibility.

    Args:
        original_params: Original parameters dict
        request_type: Type of request - 'reconciliation' or 'distillation'

    Returns:
        Adjusted parameters dict suitable for reasoning model
    """
    # Start with original parameters
    adjusted_params = original_params.copy()

    # Remove unsupported parameters
    unsupported_params = ["temperature", "max_tokens"]
    for param in unsupported_params:
        adjusted_params.pop(param, None)

    # Add max_completion_tokens if max_tokens was removed
    if "max_tokens" in original_params:
        adjusted_params["max_completion_tokens"] = original_params["max_tokens"]

    # Add reasoning model specific parameters
    reasoning_params = get_reasoning_model_parameters(request_type)
    adjusted_params.update(reasoning_params)

    logger.info(f"Adjusted parameters for reasoning model ({request_type}): {adjusted_params}")

    return adjusted_params


def make_llm_request_with_reasoning_fallback(client: Any, original_params: Dict[str, Any], request_type: str, max_retries: int = 1) -> Any:
    """
    Make LLM request with automatic fallback to reasoning model parameters.

    Args:
        client: OpenAI client instance
        original_params: Original request parameters
        request_type: Type of request - 'reconciliation' or 'distillation'
        max_retries: Maximum number of retries with adjusted parameters

    Returns:
        Response from successful LLM call

    Raises:
        ReasoningModelError: If all retry attempts fail
    """
    try:
        # First attempt with original parameters
        return client.chat.completions.create(**original_params)

    except Exception as e:
        # Check if this is a reasoning model error
        if is_reasoning_model_error(e):
            logger.info(f"Detected reasoning model error, adjusting parameters for {request_type}")

            # Adjust parameters for reasoning model
            adjusted_params = adjust_llm_params_for_reasoning_model(original_params, request_type)

            try:
                # Retry with adjusted parameters
                return client.chat.completions.create(**adjusted_params)

            except Exception as retry_error:
                raise ReasoningModelError(f"Failed to make LLM request even after adjusting for reasoning model: {retry_error}") from e
        else:
            # Re-raise original exception if not a reasoning model error
            raise e


class LLMHandlerError(Exception):
    """Base exception for LLM Handler errors."""

    pass


class LLMHandler:
    """
    Centralized handler for all LLM requests with automatic error handling and fallback.

    Handles both reconciliation and distillation requests with appropriate
    parameter adjustments and prompt generation.
    """

    def __init__(self, project_root: str = "."):
        """
        Initialize LLM Handler.

        Args:
            project_root: Project root directory for debug logging
        """
        self.project_root = project_root
        self.client = get_client()
        self.debug_logger = get_debug_logger(project_root)

    def make_reconciliation_request(self, transcript: str, candidate_symbols: List[str]) -> str:
        """
        Make a reconciliation request to mark symbols in text with backticks.

        Args:
            transcript: Original transcript text to process
            candidate_symbols: List of candidate symbols to mark

        Returns:
            Processed text with symbols marked with backticks

        Raises:
            LLMHandlerError: If the request fails after all retry attempts
        """
        if not transcript.strip() or not candidate_symbols:
            return transcript

        try:
            from .progress import reporter

            reporter.sub_step_with_progress("LLM preprocessing", "analyzing text for potential symbols", 1, 2)

            # Generate prompts for reconciliation
            system_prompt = self._create_reconciliation_system_prompt(candidate_symbols)
            user_prompt = self._create_reconciliation_user_prompt(transcript)

            # Log request for debugging
            full_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"
            self.debug_logger.log_llm_request(full_prompt, candidate_symbols)

            # Prepare request parameters
            # Prepare request parameters based on model type
            request_params = {
                "model": config.llm_model,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "max_tokens": 2000,  # Allow for longer responses with marked text
            }

            # Add parameters based on whether it's a reasoning model
            if config.is_reasoning_model:
                # For reasoning models, use max_completion_tokens and skip temperature
                request_params["max_completion_tokens"] = request_params.pop("max_tokens")
                # Note: reasoning_effort and verbosity are not yet supported by OpenAI API
            else:
                # For standard models, use temperature
                request_params["temperature"] = get_env_model_temperature()

            # Make request with reasoning model fallback (in case IS_REASONING_MODEL flag is wrong)
            response = make_llm_request_with_reasoning_fallback(client=self.client, original_params=request_params, request_type="reconciliation")

            content = response.choices[0].message.content
            if not content:
                return transcript

            # Log response for debugging
            self.debug_logger.log_llm_response(content, transcript)

            reporter.sub_step_with_progress("LLM preprocessing", "extracting marked text", 2, 2)
            return content.strip()

        except Exception as e:
            logger.warning(f"Reconciliation LLM request failed: {e}")
            # Return original text if processing fails
            return transcript

    def make_distillation_request(self, cleaned_transcript: str, compact_hints: Optional[Dict] = None, target_language: str = "en") -> Dict[str, Any]:
        """
        Make a distillation request to generate structured IR data.

        Args:
            cleaned_transcript: Cleaned transcript text for distillation
            compact_hints: Project context hints for system prompt
            target_language: Target language for the response

        Returns:
            Parsed JSON response as dictionary

        Raises:
            LLMHandlerError: If the request fails after all retry attempts
        """
        try:
            from .progress import reporter

            reporter.step_with_context("Calling the distillation model", "for final prompt generation")

            # Generate system and user prompts for distillation
            system_prompt = self._create_distillation_system_prompt(compact_hints, target_language)
            user_prompt = self._create_distillation_user_prompt(cleaned_transcript)

            # Log request for debugging (simplified for distillation)
            full_prompt = f"SYSTEM: {system_prompt}\n\nUSER: {user_prompt}"
            self.debug_logger.log_llm_request(full_prompt, [])

            # Prepare request parameters based on model type
            request_params = {
                "model": config.llm_model,
                "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                "response_format": {"type": "json_object"},
            }

            # Add parameters based on whether it's a reasoning model
            if config.is_reasoning_model:
                # For reasoning models, skip temperature
                # Note: reasoning_effort and verbosity are not yet supported by OpenAI API
                pass
            else:
                # For standard models, use temperature
                request_params["temperature"] = get_env_model_temperature()

            # Make request with reasoning model fallback (in case IS_REASONING_MODEL flag is wrong)
            response = make_llm_request_with_reasoning_fallback(client=self.client, original_params=request_params, request_type="distillation")

            reporter.complete_sub_step("Received response from distillation model")

            content = response.choices[0].message.content
            if not content:
                raise LLMHandlerError("Empty response from distillation model")

            # Log response for debugging
            self.debug_logger.log_llm_response(content, cleaned_transcript)

            # Parse JSON response
            try:
                data = json.loads(content)
                return data
            except json.JSONDecodeError as e:
                raise LLMHandlerError(f"Invalid JSON response from distillation model: {e}")

        except LLMHandlerError:
            raise
        except Exception as e:
            raise LLMHandlerError(f"Distillation LLM request failed: {e}")

    def _create_reconciliation_system_prompt(self, candidate_symbols: List[str]) -> str:
        """Create system prompt for reconciliation (symbol marking) requests."""
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

    def _create_reconciliation_user_prompt(self, transcript: str) -> str:
        """Create user prompt for reconciliation requests."""
        return f"""Mark potential code symbols in this transcript with backticks:

{transcript}

Return the complete text with symbols marked."""

    def _create_distillation_system_prompt(self, compact_hints: Optional[Dict] = None, target_language: str = "en") -> str:
        """Create system prompt for transcript distillation."""
        base_prompt = """You are an expert at distilling noisy transcripts into structured intent representations for coding agents.

Your task is to analyze a transcript and extract structured information in JSON format that follows this schema:

{
  "goal": "Primary objective or goal (required)",
  "scope_hints": ["Context and scope information"],
  "must": ["Required constraints and behaviors"],
  "must_not": ["Prohibited actions or changes"],
  "known_entities": [
    {
      "path": "file/path/or/module",
      "symbol": "function_or_class_name",
      "confidence": 0.8
    }
  ],
  "unknowns": ["Unclear or ambiguous requirements"],
  "acceptance": ["Acceptance criteria"],
  "assumptions": ["Assumptions made during analysis"]
}

Guidelines:
1. Extract the main goal clearly and concisely
2. Identify specific files, functions, classes mentioned (known_entities)
3. Separate clear requirements (must) from prohibitions (must_not)
4. Flag unclear items as unknowns rather than guessing
5. Convert uncertain facts into explicit assumptions
6. Preserve earlier goals unless explicitly overridden
7. Don't include implementation advice - focus on WHAT, not HOW
8. Use confidence scores (0.0-1.0) for known_entities based on clarity

CRITICAL: Code Identifier Preservation
- Preserve backticked identifiers verbatim; do not translate, rewrite, or pluralize them
- Treat them as code entities (functions/classes/files)
- If transcript contains code-like tokens with underscores, keep them as-is
- Code identifiers like `delete_task`, `login_handler` must remain exactly as written


LANGUAGE REQUIREMENTS:
- Produce final prompts in """

        base_prompt += "English (default)" if target_language == "en" else "the source language"
        base_prompt += """
- If source is not English and target is English, translate narrative text only, and preserve backticked identifiers verbatim
- Never translate code identifiers, file paths, or technical symbols

"""

        if compact_hints:
            hint_text = self._format_compact_hints(compact_hints)
            base_prompt += f"\nProject Context:\n{hint_text}\n"

        base_prompt += "\nRespond only with valid JSON matching the schema above."

        return base_prompt

    def _create_distillation_user_prompt(self, cleaned_transcript: str) -> str:
        """Create user prompt for distillation requests."""
        return f"""Analyze this transcript and extract structured intent information:

TRANSCRIPT:
{cleaned_transcript}

Extract the information into the JSON schema format as instructed."""

    def _format_compact_hints(self, hints: Dict) -> str:
        """Format compact hints for inclusion in system prompt."""
        lines = []

        if hints.get("top_dirs"):
            lines.append("Top-level directories:")
            for directory in hints["top_dirs"][:5]:  # Top 5
                lines.append(f"  - {directory}")

        if hints.get("sample_files"):
            lines.append("Sample files:")
            for file_path in hints["sample_files"][:10]:  # Top 10
                lines.append(f"  - {file_path}")

        if hints.get("known_symbols"):
            lines.append("Known symbols:")
            symbol_list = ", ".join(hints["known_symbols"][:20])  # Top 20
            lines.append(f"  {symbol_list}")

        return "\n".join(lines)


# Global LLM handler instance
_llm_handler: Optional[LLMHandler] = None


def get_llm_handler(project_root: str = ".") -> LLMHandler:
    """
    Get or create global LLM handler instance.

    Args:
        project_root: Project root directory

    Returns:
        LLMHandler instance
    """
    global _llm_handler
    if _llm_handler is None or _llm_handler.project_root != project_root:
        _llm_handler = LLMHandler(project_root)
    return _llm_handler
