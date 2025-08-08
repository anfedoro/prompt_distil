"""
Symbol reconciliation and text processing module.

This module provides functionality for matching text mentions to known symbols
in the project, including fuzzy matching, alias generation, and text normalization.
Supports multilingual lexicon-based normalization for improved identifier reconciliation.
"""

import re
from typing import Dict, List, Optional, Tuple

from rapidfuzz import fuzz

from .timing import timer

# Reconciliation thresholds
CONTEXT_THRESHOLD = 0.75
GENERAL_THRESHOLD = 0.8


def normalize_text(text: str) -> str:
    """
    Normalize text for matching by removing punctuation, lowercasing, etc.

    Args:
        text: Input text to normalize

    Returns:
        Normalized text
    """
    # Lowercase
    text = text.lower()

    # Remove punctuation except underscores and hyphens
    text = re.sub(r"[^\w\s\-_]", " ", text)

    # Collapse multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Strip articles and common words
    articles = {"a", "an", "the", "this", "that", "these", "those"}
    words = text.split()
    words = [w for w in words if w not in articles]

    return " ".join(words).strip()


@timer
def fuzzy_match_symbol(query: str, symbol_aliases: List[str], threshold: float = GENERAL_THRESHOLD) -> Optional[float]:
    """
    Perform fuzzy matching between query and symbol aliases.

    Args:
        query: The query text to match
        symbol_aliases: List of aliases for a symbol
        threshold: Minimum similarity threshold

    Returns:
        Best match score if above threshold, None otherwise
    """
    best_score = 0.0

    normalized_query = normalize_text(query)

    for alias in symbol_aliases:
        normalized_alias = normalize_text(alias)

        # Exact match gets perfect score
        if normalized_query == normalized_alias:
            return 1.0

        # Fuzzy match using rapidfuzz
        score = fuzz.ratio(normalized_query, normalized_alias) / 100.0

        # Also try substring matching
        if normalized_query in normalized_alias or normalized_alias in normalized_query:
            substring_score = len(normalized_query) / max(len(normalized_query), len(normalized_alias))
            score = max(score, substring_score)

        best_score = max(best_score, score)

    return best_score if best_score >= threshold else None


@timer
def reconcile_text(text: str, project_root: str = ".", asr_language: str = "auto") -> Tuple[str, List[str], List[str], List[str]]:
    """
    Reconcile text by finding and replacing symbol mentions with canonical backticked names.
    Uses LLM to identify and mark symbols, then applies basic fuzzy matching.

    Args:
        text: Input text to reconcile
        project_root: Root directory for loading symbol cache
        asr_language: Language detected by ASR (unused, kept for compatibility)

    Returns:
        Tuple of (reconciled_text, list_of_matched_symbols, list_of_unknown_mentions, unresolved_terms)
    """
    from .progress import reporter
    from .surface import load_cache

    # Step 1: Load symbol cache
    reporter.sub_step("Loading symbol cache", 1, 3)

    cache = load_cache(project_root)
    if not cache:
        return text, [], [], []

    known_symbols = {s["name"]: s for s in cache.get("symbols", [])}
    if not known_symbols:
        return text, [], [], []

    # Step 2: Use LLM to identify and mark symbols
    reporter.sub_step("Invoking LLM for symbol identification", 2, 3)
    reconciled_text, matched_symbols, unknown_mentions = _process_llm_mode(text, known_symbols, project_root)

    # Step 3: Return results
    reporter.sub_step("Finalizing reconciliation results", 3, 3)
    unresolved_terms = []  # No longer needed as LLM handles everything

    return reconciled_text, matched_symbols, unknown_mentions, unresolved_terms


def _process_llm_mode(text: str, known_symbols: Dict[str, Dict], project_root: str) -> Tuple[str, List[str], List[str]]:
    """
    Process text using LLM to identify and mark symbols.

    Args:
        text: Input text to reconcile
        known_symbols: Dictionary of known symbols from cache
        project_root: Project root directory

    Returns:
        Tuple of (reconciled_text, matched_symbols, unknown_mentions)
    """
    from .debug_log import get_debug_logger
    from .llm_map import llm_preprocess_text

    # Initialize debug logger
    debug_logger = get_debug_logger(project_root)

    # Use LLM to identify and mark symbols
    all_symbols = list(known_symbols.keys())

    if not all_symbols:
        return text, [], []

    # Call LLM to mark symbols with backticks
    preprocessed_text = llm_preprocess_text(text, all_symbols)

    # Extract matched symbols from LLM output
    matched_symbols, unknown_mentions = _extract_symbols_from_llm_output(preprocessed_text, known_symbols)

    # Log reconciliation summary
    debug_logger.log_reconciliation_summary(text, preprocessed_text, matched_symbols, unknown_mentions, [])

    return preprocessed_text, matched_symbols, unknown_mentions


def _extract_symbols_from_llm_output(text: str, known_symbols: Dict[str, Dict]) -> Tuple[List[str], List[str]]:
    """
    Extract matched and unknown symbols from LLM-processed text with backticks.

    Args:
        text: Text with LLM-marked symbols (backticks)
        known_symbols: Dictionary of known symbols from cache

    Returns:
        Tuple of (matched_symbols, unknown_mentions)
    """
    import re

    matched_symbols = []
    unknown_mentions = []

    # Extract backtick-marked phrases
    backtick_pattern = r"`([^`]+)`"
    marked_phrases = re.findall(backtick_pattern, text)

    for phrase in marked_phrases:
        # Direct match with known symbols
        if phrase in known_symbols:
            if phrase not in matched_symbols:
                matched_symbols.append(phrase)
        else:
            # Simple fuzzy matching for close matches
            best_match = None
            best_score = 0.0

            for symbol_name in known_symbols:
                # Basic fuzzy matching
                score = fuzzy_match_symbol(phrase, [symbol_name], threshold=0.8)
                if score and score > best_score:
                    best_score = score
                    best_match = symbol_name

            if best_match and best_match not in matched_symbols:
                matched_symbols.append(best_match)
            else:
                if phrase not in unknown_mentions:
                    unknown_mentions.append(phrase)

    return matched_symbols, unknown_mentions
