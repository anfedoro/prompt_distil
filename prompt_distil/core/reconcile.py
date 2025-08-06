"""
Symbol reconciliation and text processing module.

This module provides functionality for matching text mentions to known symbols
in the project, including fuzzy matching, alias generation, and text normalization.
Supports multilingual lexicon-based normalization for improved identifier reconciliation.
"""

import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from .lexicon import (
    generate_lexicon_aware_aliases,
    generate_stemmed_aliases,
    get_effective_language,
    normalize_phrase_with_lexicon,
    stem_tokens,
)

# Reconciliation thresholds
CONTEXT_THRESHOLD = 0.75
GENERAL_THRESHOLD = 0.8


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    # Insert underscore before uppercase letters that follow lowercase letters
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Insert underscore before uppercase letters that follow lowercase letters or digits
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def generate_aliases(symbol_name: str, symbol_kind: str, lang: str = "en", project_root: Union[str, Path] = ".") -> List[str]:
    """
    Generate aliases for a symbol to improve matching with lexicon awareness.

    Args:
        symbol_name: The canonical symbol name
        symbol_kind: The kind of symbol (function, class, etc.)
        lang: Language code for lexicon processing
        project_root: Project root for loading lexicon overrides

    Returns:
        List of possible aliases for the symbol
    """
    aliases = [symbol_name]

    # Add snake_case version if it's CamelCase
    if symbol_name != symbol_name.lower() and "_" not in symbol_name:
        snake_version = camel_to_snake(symbol_name)
        aliases.append(snake_version)

    # Add spaced versions
    if "_" in symbol_name:
        # Convert underscores to spaces and hyphens
        spaced = symbol_name.replace("_", " ")
        hyphenated = symbol_name.replace("_", "-")
        aliases.extend([spaced, hyphenated])

    # For CamelCase, add various formats
    if symbol_name != symbol_name.lower() and "_" not in symbol_name:
        # Add camelCase version (first letter lowercase)
        camel_case = symbol_name[0].lower() + symbol_name[1:] if len(symbol_name) > 1 else symbol_name.lower()
        aliases.append(camel_case)

        # Add handler-style aliases for classes
        if symbol_kind == "class" and not symbol_name.lower().endswith("handler"):
            aliases.append(f"{symbol_name}Handler")
            aliases.append(f"{camel_case}_handler")
            aliases.append(f"{symbol_name.lower()} handler")

    # Add lexicon-aware aliases if not English
    if lang != "en":
        lexicon_aliases = generate_lexicon_aware_aliases(symbol_name, lang, project_root)
        aliases.extend(lexicon_aliases)

        # Also try with spaced version
        if "_" in symbol_name:
            spaced_aliases = generate_lexicon_aware_aliases(symbol_name.replace("_", " "), lang, project_root)
            aliases.extend(spaced_aliases)

    # Remove duplicates while preserving order
    seen = set()
    unique_aliases = []
    for alias in aliases:
        if alias and alias not in seen:
            seen.add(alias)
            unique_aliases.append(alias)

    return unique_aliases


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


def extract_ngrams(text: str, min_len: int = 1, max_len: int = 3) -> List[str]:
    """
    Extract n-grams from text for fuzzy matching.

    Args:
        text: Input text
        min_len: Minimum n-gram length in words
        max_len: Maximum n-gram length in words

    Returns:
        List of n-grams
    """
    words = text.split()
    ngrams = []

    for n in range(min_len, min_len + max_len):
        for i in range(len(words) - n + 1):
            ngram = " ".join(words[i : i + n])
            ngrams.append(ngram)

    return ngrams


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

        # Fuzzy match using SequenceMatcher
        score = SequenceMatcher(None, normalized_query, normalized_alias).ratio()

        # Also try substring matching
        if normalized_query in normalized_alias or normalized_alias in normalized_query:
            substring_score = len(normalized_query) / max(len(normalized_query), len(normalized_alias))
            score = max(score, substring_score)

        best_score = max(best_score, score)

    return best_score if best_score >= threshold else None


def apply_context_rules(text: str, known_symbols: Dict[str, Dict], lang: str = "en", project_root: Union[str, Path] = ".") -> List[Tuple[str, str, float]]:
    """
    Apply context-specific matching rules.

    Args:
        text: Input text to analyze
        known_symbols: Dictionary of known symbols

    Returns:
        List of (original_text, symbol_name, confidence) tuples
    """
    matches = []

    # Context rule: "тест на|test for" with lower threshold
    test_pattern = r"(тест на|test for)\s+(?P<phrase>.+?)\b"
    for match in re.finditer(test_pattern, text, re.IGNORECASE):
        phrase = match.group("phrase").strip()

        # Try matching with lower threshold for test contexts
        for symbol_name, symbol_info in known_symbols.items():
            aliases = generate_aliases(symbol_name, symbol_info.get("kind", "unknown"), lang, project_root)
            score = fuzzy_match_symbol(phrase, aliases, threshold=CONTEXT_THRESHOLD)
            if score:
                matches.append((match.group(0), symbol_name, score))
                break

    # Handle "XTest" patterns - if symbol doesn't exist but "X" does
    test_suffix_pattern = r"\b(\w+)Test\b"
    for match in re.finditer(test_suffix_pattern, text):
        test_name = match.group(1)
        base_name = test_name.rstrip("Test")

        # Check if base symbol exists
        potential_names = [base_name, camel_to_snake(base_name), f"{base_name.lower()}_task"]
        for potential in potential_names:
            if potential in known_symbols:
                matches.append((match.group(0), potential, 0.9))
                break

    return matches


def reconcile_text(
    text: str, project_root: str = ".", asr_language: str = "auto", lex_mode: Literal["rules", "llm", "hybrid"] = "hybrid"
) -> Tuple[str, List[str], List[str], List[str], List[str]]:
    """
    Reconcile text by finding and replacing symbol mentions with canonical backticked names.
    Only backticks symbols that exist in the project cache. Uses lexicon for language-aware normalization
    with optional stemming and LLM fallback.

    Args:
        text: Input text to reconcile
        project_root: Root directory for loading symbol cache
        asr_language: Language detected by ASR (or "auto")
        lex_mode: Lexicon mode ("rules", "llm", "hybrid")

    Returns:
        Tuple of (reconciled_text, list_of_matched_symbols, list_of_unknown_mentions, lexicon_hits, unresolved_terms)
    """
    from .surface import load_cache

    # Determine effective language for lexicon processing
    effective_lang, _ = get_effective_language(asr_language, text, project_root)

    # Load symbol cache from project root
    cache = load_cache(project_root)
    if not cache:
        return text, [], [], [], []

    known_symbols = {s["name"]: s for s in cache.get("symbols", [])}
    if not known_symbols:
        return text, [], [], [], []

    matched_symbols = []
    unknown_mentions = []
    lexicon_hits = []
    unresolved_terms = []
    reconciled_text = text

    # Rules-based processing
    if lex_mode in {"rules", "hybrid"}:
        matched_symbols, unknown_mentions, lexicon_hits, reconciled_text = _process_rules_based(text, known_symbols, effective_lang, project_root)

        # Collect unresolved terms for potential LLM processing
        if lex_mode == "hybrid":
            unresolved_terms = _extract_unresolved_terms(text, matched_symbols)

    # LLM-based processing
    if lex_mode in {"llm", "hybrid"} and (lex_mode == "llm" or unresolved_terms):
        candidate_symbols = list(known_symbols.keys())
        llm_matched, llm_unknown = _process_llm_based(text, candidate_symbols, known_symbols, unresolved_terms if lex_mode == "hybrid" else None)

        # Merge results
        for symbol in llm_matched:
            if symbol not in matched_symbols:
                matched_symbols.append(symbol)
                # Replace in text
                reconciled_text = reconciled_text.replace(symbol.replace("_", " "), f"`{symbol}`")
                reconciled_text = reconciled_text.replace(symbol, f"`{symbol}`")

        unknown_mentions.extend(llm_unknown)

    return reconciled_text, matched_symbols, unknown_mentions, lexicon_hits, unresolved_terms


def _process_rules_based(text: str, known_symbols: Dict[str, Dict], effective_lang: str, project_root: str) -> Tuple[List[str], List[str], List[str], str]:
    """Process text using rules-based approach with stemming."""
    matched_symbols = []
    unknown_mentions = []
    lexicon_hits = []
    reconciled_text = text

    # Apply lexicon normalization to the entire text first
    normalized_text, text_lexicon_hits, _ = normalize_phrase_with_lexicon(text, effective_lang, project_root)
    lexicon_hits.extend(text_lexicon_hits)

    # Apply context rules first (on both original and normalized text)
    context_matches = apply_context_rules(text, known_symbols, effective_lang, project_root)
    context_matches_norm = apply_context_rules(normalized_text, known_symbols, effective_lang, project_root)

    # Combine context matches
    all_context_matches = context_matches + context_matches_norm

    # Process context matches - only backtick if symbol exists in cache
    for original, symbol_name, confidence in all_context_matches:
        if symbol_name in known_symbols and symbol_name not in matched_symbols:
            matched_symbols.append(symbol_name)
            # Replace with backticked version
            reconciled_text = reconciled_text.replace(original, f"`{symbol_name}`")

    # Extract n-grams for general fuzzy matching with stemming
    original_tokens = normalize_text(text).split()
    lexicon_tokens = normalize_text(normalized_text).split()

    # Apply stemming to tokens
    stemmed_original = stem_tokens(original_tokens, effective_lang)
    stemmed_lexicon = stem_tokens(lexicon_tokens, effective_lang)

    # Generate n-grams from stemmed tokens
    ngrams = extract_ngrams(" ".join(stemmed_original))
    ngrams.extend(extract_ngrams(" ".join(stemmed_lexicon)))

    # Add original n-grams for fallback
    ngrams.extend(extract_ngrams(" ".join(original_tokens)))
    ngrams.extend(extract_ngrams(" ".join(lexicon_tokens)))

    # Remove duplicates
    ngrams = list(set(ngrams))

    # Track replacements to avoid conflicts
    replacements = []

    for ngram in ngrams:
        if len(ngram.strip()) < 2:  # Skip very short ngrams
            continue

        best_match = None
        best_score = 0.0

        # Only consider symbols that exist in the cache
        for symbol_name, symbol_info in known_symbols.items():
            # Generate aliases with stemming support
            aliases = generate_aliases(symbol_name, symbol_info.get("kind", "unknown"), effective_lang, project_root)
            aliases.extend(generate_stemmed_aliases(symbol_name, effective_lang))

            score = fuzzy_match_symbol(ngram, aliases)

            if score and score > best_score:
                best_score = score
                best_match = symbol_name

        if best_match and best_match in known_symbols and best_match not in matched_symbols:
            # Find original case version in text for replacement
            original_ngram = _find_original_case(text, ngram)
            if original_ngram and original_ngram not in [r[0] for r in replacements]:
                replacements.append((original_ngram, f"`{best_match}`"))
                matched_symbols.append(best_match)
        elif best_score > 0.6:  # Potential match but not in cache
            original_ngram = _find_original_case(text, ngram)
            if original_ngram and original_ngram not in unknown_mentions:
                unknown_mentions.append(original_ngram)

    # Apply replacements, longest first to avoid partial replacements
    replacements.sort(key=lambda x: len(x[0]), reverse=True)
    for original, replacement in replacements:
        if original in reconciled_text:
            reconciled_text = reconciled_text.replace(original, replacement)

    return matched_symbols, unknown_mentions, lexicon_hits, reconciled_text


def _process_llm_based(
    text: str, candidate_symbols: List[str], known_symbols: Dict[str, Dict], unresolved_terms: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """Process text using LLM-based approach."""
    try:
        from .llm_map import filter_high_confidence_mappings, llm_map_symbols, validate_mappings_against_cache

        # Get LLM mappings
        mappings = llm_map_symbols(text, candidate_symbols)

        # Filter high confidence mappings
        high_conf_mappings = filter_high_confidence_mappings(mappings, min_confidence=0.8)

        # Validate against cache
        cache_symbols = set(known_symbols.keys())
        valid_mappings = validate_mappings_against_cache(high_conf_mappings, cache_symbols)

        # Extract matched symbols
        matched_symbols = [m.symbol for m in valid_mappings]

        # Extract unknown mentions (phrases that didn't map with high confidence)
        unknown_mentions = []
        for mapping in mappings:
            if mapping.confidence < 0.8 or mapping.symbol not in cache_symbols:
                if mapping.text not in unknown_mentions:
                    unknown_mentions.append(mapping.text)

        return matched_symbols, unknown_mentions

    except Exception:
        # If LLM processing fails, return empty results
        return [], []


def _extract_unresolved_terms(text: str, matched_symbols: List[str]) -> List[str]:
    """Extract terms that look code-like but weren't resolved by rules."""
    # Simple heuristic: find words with underscores or CamelCase that weren't matched
    import re

    # Find potential code terms
    code_patterns = [
        r"\b[a-z]+_[a-z_]+\b",  # snake_case
        r"\b[A-Z][a-z]+[A-Z][a-zA-Z]*\b",  # CamelCase
        r"\b[a-z]+[A-Z][a-zA-Z]*\b",  # camelCase
    ]

    unresolved = []
    for pattern in code_patterns:
        matches = re.findall(pattern, text)
        for match in matches:
            # Check if this term was already matched
            if match not in matched_symbols and f"`{match}`" not in text:
                if match not in unresolved:
                    unresolved.append(match)

    return unresolved[:10]  # Limit to prevent overflow


def _find_original_case(text: str, normalized_ngram: str) -> Optional[str]:
    """
    Find the original case version of a normalized n-gram in the text.

    Args:
        text: Original text
        normalized_ngram: Normalized n-gram to find

    Returns:
        Original case version if found, None otherwise
    """
    # Split normalized ngram into words
    norm_words = normalized_ngram.split()
    if not norm_words:
        return None

    # Simple approach: look for each word separately and try to find the best match
    # This avoids complex regex that might fail
    for word in norm_words:
        # Create a simple case-insensitive pattern for each word
        pattern = r"\b" + re.escape(word) + r"\b"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(0)

    # If no individual word matches, return None
    return None
