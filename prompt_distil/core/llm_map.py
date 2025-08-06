"""
LLM-backed symbol mapping functionality.

This module provides LLM-assisted mapping of transcript phrases to code symbols
when deterministic rule-based matching fails. Uses OpenAI API with strict constraints
to map only to symbols that exist in the project cache.
"""

import json
from typing import Dict, List

from .config import config, get_client


class LLMMapError(Exception):
    """Raised when LLM symbol mapping fails."""

    pass


class Mapping:
    """Represents a mapping from text phrase to symbol."""

    def __init__(self, text: str, symbol: str, confidence: float, reason: str = ""):
        self.text = text
        self.symbol = symbol
        self.confidence = confidence
        self.reason = reason

    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {"text": self.text, "symbol": self.symbol, "confidence": self.confidence, "reason": self.reason}

    @classmethod
    def from_dict(cls, data: Dict) -> "Mapping":
        """Create from dictionary representation."""
        return cls(text=data.get("text", ""), symbol=data.get("symbol", ""), confidence=float(data.get("confidence", 0.0)), reason=data.get("reason", ""))


def llm_map_symbols(transcript: str, candidate_symbols: List[str], max_candidates: int = 200) -> List[Mapping]:
    """
    Map phrases in transcript to code symbols using LLM.

    Args:
        transcript: Original transcript text to analyze
        candidate_symbols: List of symbol names that exist in the project cache
        max_candidates: Maximum number of candidate symbols to include in prompt

    Returns:
        List of Mapping objects with confidence scores

    Raises:
        LLMMapError: If LLM call fails or returns invalid response
    """
    if not transcript.strip() or not candidate_symbols:
        return []

    # Limit candidate symbols to prevent prompt overflow
    limited_candidates = candidate_symbols[:max_candidates]

    # Create system prompt
    system_prompt = _create_llm_system_prompt(limited_candidates)

    # Create user prompt
    user_prompt = _create_llm_user_prompt(transcript)

    try:
        from .progress import reporter

        reporter.step_with_context("Calling the model", "for reconciliation")
        client = get_client()
        response = client.chat.completions.create(
            model=config.distil_model,
            messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,  # Low temperature for consistent mapping
            max_tokens=1000,  # Limit response size
        )

        content = response.choices[0].message.content
        if not content:
            return []

        # Parse JSON response
        try:
            data = json.loads(content)
            mappings = []

            for mapping_data in data.get("mappings", []):
                try:
                    mapping = Mapping.from_dict(mapping_data)

                    # Validate mapping
                    if mapping.symbol in limited_candidates and mapping.confidence >= 0.0 and mapping.confidence <= 1.0 and mapping.text.strip():
                        mappings.append(mapping)

                except (ValueError, KeyError):
                    # Skip invalid mappings
                    continue

            return mappings

        except json.JSONDecodeError as e:
            raise LLMMapError(f"Invalid JSON response from LLM: {e}")

    except Exception as e:
        raise LLMMapError(f"LLM symbol mapping failed: {e}")


def _create_llm_system_prompt(candidate_symbols: List[str]) -> str:
    """Create system prompt for LLM symbol mapping."""
    symbols_text = ", ".join(f"`{symbol}`" for symbol in candidate_symbols[:50])  # Show first 50
    if len(candidate_symbols) > 50:
        symbols_text += f" ... and {len(candidate_symbols) - 50} more"

    return f"""You are an expert at mapping natural language phrases to code symbols.

Your task is to identify phrases in a transcript that likely refer to code symbols and map them to the closest matching symbol from the allowed list.

STRICT CONSTRAINTS:
1. You MAY ONLY choose symbols from this exact list: {symbols_text}
2. If you cannot find a good match (confidence < 0.7), do not include it
3. Preserve any existing backticked identifiers as-is
4. Focus on phrases that sound like function names, class names, or variable names
5. Consider common variations: snake_case ↔ CamelCase, plurals, verb forms

Return JSON in this exact format:
{{
  "mappings": [
    {{
      "text": "exact phrase from transcript",
      "symbol": "matching_symbol_from_list",
      "confidence": 0.85,
      "reason": "brief explanation"
    }}
  ]
}}

If no mappings are found, return: {{"mappings": []}}"""


def _create_llm_user_prompt(transcript: str) -> str:
    """Create user prompt containing the transcript to analyze."""
    return f"""Analyze this transcript and find phrases that likely refer to code symbols:

TRANSCRIPT:
{transcript}

Map any code-like phrases to symbols from the allowed list. Return JSON only."""


def filter_high_confidence_mappings(mappings: List[Mapping], min_confidence: float = 0.8) -> List[Mapping]:
    """
    Filter mappings to keep only high-confidence ones.

    Args:
        mappings: List of mappings to filter
        min_confidence: Minimum confidence threshold

    Returns:
        Filtered list of high-confidence mappings
    """
    return [m for m in mappings if m.confidence >= min_confidence]


def validate_mappings_against_cache(mappings: List[Mapping], cache_symbols: set) -> List[Mapping]:
    """
    Validate that all mapped symbols exist in the cache.

    Args:
        mappings: List of mappings to validate
        cache_symbols: Set of symbol names that exist in the project cache

    Returns:
        Filtered list containing only mappings to symbols that exist in cache
    """
    return [m for m in mappings if m.symbol in cache_symbols]


def merge_rule_and_llm_mappings(rule_mappings: List[str], llm_mappings: List[Mapping]) -> tuple[List[str], List[Mapping]]:
    """
    Merge rule-based and LLM mappings, avoiding duplicates.

    Args:
        rule_mappings: List of symbol names found by rule-based matching
        llm_mappings: List of LLM mappings

    Returns:
        Tuple of (final_symbols, llm_mappings_used) where llm_mappings_used
        contains only the LLM mappings that weren't already found by rules
    """
    rule_set = set(rule_mappings)
    unique_llm_mappings = [m for m in llm_mappings if m.symbol not in rule_set]

    final_symbols = rule_mappings + [m.symbol for m in unique_llm_mappings]

    return final_symbols, unique_llm_mappings
