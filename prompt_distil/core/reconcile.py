"""
Symbol reconciliation and text processing module.

This module provides functionality for matching text mentions to known symbols
in the project, including fuzzy matching, alias generation, and text normalization.
Supports multilingual lexicon-based normalization for improved identifier reconciliation.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

from rapidfuzz import fuzz

from .lexicon import (
    generate_lexicon_aware_aliases,
    generate_stemmed_aliases,
    get_effective_language,
    normalize_phrase_with_lexicon,
    stem_tokens,
)
from .timing import timer

# Reconciliation thresholds
CONTEXT_THRESHOLD = 0.75
GENERAL_THRESHOLD = 0.8


def camel_to_snake(name: str) -> str:
    """Convert CamelCase to snake_case."""
    # Insert underscore before uppercase letters that follow lowercase letters
    s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    # Insert underscore before uppercase letters that follow lowercase letters or digits
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()


def _filter_relevant_symbols(text: str, known_symbols: Dict[str, Dict], max_symbols: int = 50) -> List[str]:
    """
    Filter project symbols to most relevant ones for LLM processing.

    Args:
        text: Input text to analyze
        known_symbols: Dictionary of all known symbols
        max_symbols: Maximum number of symbols to return

    Returns:
        List of most relevant symbol names
    """
    # Extract potential identifier-like words from text
    identifier_words = re.findall(r"\b[a-zA-Z_][a-zA-Z0-9_]*\b", text.lower())

    # Score symbols based on similarity to words in text
    symbol_scores = {}

    for symbol_name in known_symbols.keys():
        score = 0
        symbol_lower = symbol_name.lower()

        # Direct word matches get highest score
        for word in identifier_words:
            if word in symbol_lower or symbol_lower in word:
                score += 10

            # Partial matches (for snake_case vs camelCase)
            word_parts = word.split("_")
            symbol_parts = symbol_lower.split("_")

            for word_part in word_parts:
                for symbol_part in symbol_parts:
                    if len(word_part) > 2 and len(symbol_part) > 2:
                        if word_part in symbol_part or symbol_part in word_part:
                            score += 3

        # Bonus for commonly referenced symbols
        if any(keyword in symbol_lower for keyword in ["test", "main", "init", "config", "handler"]):
            score += 1

        if score > 0:
            symbol_scores[symbol_name] = score

    # Return top scoring symbols
    sorted_symbols = sorted(symbol_scores.items(), key=lambda x: x[1], reverse=True)
    return [symbol for symbol, _ in sorted_symbols[:max_symbols]]


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
    return _generate_aliases_cached(symbol_name, symbol_kind, lang, str(project_root))


@lru_cache(maxsize=512)
def _generate_aliases_cached(symbol_name: str, symbol_kind: str, lang: str = "en", project_root_str: str = ".") -> List[str]:
    """Cached implementation of generate_aliases."""
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
        lexicon_aliases = generate_lexicon_aware_aliases(symbol_name, lang, project_root_str)
        aliases.extend(lexicon_aliases)

        # Also try with spaced version
        if "_" in symbol_name:
            spaced_aliases = generate_lexicon_aware_aliases(symbol_name.replace("_", " "), lang, project_root_str)
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


@timer
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
    from .progress import reporter
    from .surface import load_cache

    # Step 1: Language detection and cache loading
    reporter.sub_step("Detecting language and loading symbol cache", 1, 5)

    # Determine effective language for lexicon processing
    effective_lang, lang_meta = get_effective_language(asr_language, text, project_root)

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

    # Step 2: Process based on mode with explicit model calls
    if lex_mode == "hybrid":
        # New optimized hybrid mode: LLM first, then rules on marked content
        reporter.sub_step("Invoking reconciliation model and performing n-gram analysis", 2, 5)
        reconciled_text, matched_symbols, unknown_mentions, lexicon_hits = _process_llm_first_hybrid(text, known_symbols, effective_lang, project_root)
    elif lex_mode == "rules":
        # Traditional rules-only processing
        reporter.sub_step("Performing rules-based n-gram matching", 2, 5)
        matched_symbols, unknown_mentions, lexicon_hits, reconciled_text = _process_rules_based(text, known_symbols, effective_lang, project_root)
        reporter.complete_sub_step("Completed rules-based n-gram matching")
    elif lex_mode == "llm":
        # LLM-only processing - remove any existing backticks first
        reporter.sub_step("Invoking reconciliation model for symbol mapping", 2, 5)

        # Remove all backticks from text before LLM processing
        import re

        clean_text = re.sub(r"`([^`]+)`", r"\1", text)

        candidate_symbols = list(known_symbols.keys())
        llm_matched, llm_unknown = _process_llm_based(clean_text, candidate_symbols, known_symbols, None)
        reporter.complete_sub_step("Called reconciliation model for symbol mapping")

        matched_symbols = llm_matched
        unknown_mentions = llm_unknown

        # In LLM-only mode, do NOT add backticks to the reconciled text
        # Keep the text clean without backticks for the distillation model
        reconciled_text = clean_text

    return reconciled_text, matched_symbols, unknown_mentions, lexicon_hits, unresolved_terms


def _process_rules_based(text: str, known_symbols: Dict[str, Dict], effective_lang: str, project_root: str) -> Tuple[List[str], List[str], List[str], str]:
    """Process text using rules-based approach with stemming."""
    from .progress import reporter

    matched_symbols = []
    unknown_mentions = []
    lexicon_hits = []
    reconciled_text = text

    # Sub-step 1: Apply lexicon normalization
    reporter.sub_step_with_progress("Rules-based n-gram matching", "applying lexicon normalization", 1, 5)
    normalized_text, text_lexicon_hits, _ = normalize_phrase_with_lexicon(text, effective_lang, project_root)
    lexicon_hits.extend(text_lexicon_hits)

    # Sub-step 2: Apply context rules
    reporter.sub_step_with_progress("Rules-based n-gram matching", "applying context rules", 2, 5)
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

    # Sub-step 3: Extract and process n-grams
    reporter.sub_step_with_progress("Rules-based n-gram matching", "extracting n-grams for symbol search", 3, 5)
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

    # Sub-step 4: Perform fuzzy matching on n-grams
    reporter.sub_step_with_progress("Rules-based n-gram matching", f"performing fuzzy matching on {len(ngrams)} n-grams", 4, 5)

    # Initialize debug logger for n-gram comparisons
    from .debug_log import get_debug_logger

    debug_logger = get_debug_logger(project_root)

    # Track replacements to avoid conflicts
    replacements = []

    for i, ngram in enumerate(ngrams):
        if len(ngram.strip()) < 2:  # Skip very short ngrams
            continue

        # Report progress for large ngram sets
        if len(ngrams) > 50 and i % 10 == 0:
            reporter.sub_step_with_progress("Rules-based n-gram matching", f"processing n-grams ({i + 1}/{len(ngrams)})", 4, 5)

        best_match = None
        best_score = 0.0
        ngram_matches = []  # Track all matches for this ngram

        # Only consider symbols that exist in the cache
        for symbol_name, symbol_info in known_symbols.items():
            # Generate aliases with stemming support
            aliases = generate_aliases(symbol_name, symbol_info.get("kind", "unknown"), effective_lang, project_root)
            aliases.extend(generate_stemmed_aliases(symbol_name, effective_lang))

            score = fuzzy_match_symbol(ngram, aliases)

            # Record match details for debugging
            match_result = {
                "ngram": ngram,
                "symbol": symbol_name,
                "score": score if score is not None else 0.0,
                "aliases_used": aliases,
                "matched": score is not None and score > 0.8,
            }
            ngram_matches.append(match_result)

            if score and score > best_score:
                best_score = score
                best_match = symbol_name

        # Log n-gram comparison details for debugging
        if ngram_matches:
            debug_logger.log_ngram_comparison("rules_based_ngram_matching", [ngram], "multiple_symbols", [], ngram_matches)

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

    # Sub-step 5: Apply symbol replacements
    reporter.sub_step_with_progress("Rules-based n-gram matching", f"applying {len(replacements)} symbol replacements", 5, 5)

    # Apply replacements, longest first to avoid partial replacements
    replacements.sort(key=lambda x: len(x[0]), reverse=True)
    for original, replacement in replacements:
        if original in reconciled_text:
            reconciled_text = reconciled_text.replace(original, replacement)

    return matched_symbols, unknown_mentions, lexicon_hits, reconciled_text


def _process_llm_first_hybrid(text: str, known_symbols: Dict[str, Dict], effective_lang: str, project_root: str) -> Tuple[str, List[str], List[str], List[str]]:
    """
    Process text using LLM-first approach for optimized hybrid mode.

    Steps:
    1. Filter relevant symbols for LLM
    2. Use LLM to mark potential symbols with backticks
    3. Apply rules only to backtick-marked content

    Returns:
        Tuple of (processed_text, matched_symbols, unknown_mentions, lexicon_hits)
    """
    from .debug_log import get_debug_logger
    from .llm_map import llm_preprocess_text
    from .progress import reporter

    # Initialize debug logger
    debug_logger = get_debug_logger(project_root)

    # Step 1: Filter relevant symbols for LLM processing
    reporter.sub_step_with_progress("Reconciliation model processing", "filtering relevant project symbols", 1, 4)
    original_symbols = list(known_symbols.keys())
    relevant_symbols = _filter_relevant_symbols(text, known_symbols, max_symbols=50)

    # Log symbol filtering details
    debug_logger.log_symbol_filtering(
        "hybrid_symbol_filtering", original_symbols, relevant_symbols, {"max_symbols": 50, "text": text, "effective_lang": effective_lang}
    )

    reporter.complete_sub_step("Filtered relevant project symbols for reconciliation")

    if not relevant_symbols:
        # No relevant symbols found, fall back to original text
        return text, [], [], []

    # Step 2: Use LLM to preprocess and mark symbols
    reporter.sub_step_with_progress("Reconciliation model processing", "calling LLM for symbol identification", 2, 4)
    preprocessed_text = llm_preprocess_text(text, relevant_symbols)
    reporter.complete_sub_step("Called reconciliation model for symbol identification")

    # Step 3: Apply rules selectively to marked content with n-gram analysis
    reporter.sub_step_with_progress("Reconciliation model processing", "performing n-gram search on identified symbols", 3, 4)
    matched_symbols, unknown_mentions, lexicon_hits, reconciled_text = _process_marked_text_with_rules(
        preprocessed_text, known_symbols, effective_lang, project_root
    )
    reporter.complete_sub_step("Performed n-gram search on identified symbols")

    # Step 4: Finalize reconciliation results
    reporter.sub_step_with_progress("Reconciliation model processing", "finalizing symbol reconciliation", 4, 4)

    # Log reconciliation summary
    debug_logger.log_reconciliation_summary(text, reconciled_text, matched_symbols, unknown_mentions, lexicon_hits)

    reporter.complete_sub_step("Finalized symbol reconciliation")

    return reconciled_text, matched_symbols, unknown_mentions, lexicon_hits


def _process_marked_text_with_rules(
    text: str, known_symbols: Dict[str, Dict], effective_lang: str, project_root: str
) -> Tuple[List[str], List[str], List[str], str]:
    """
    Process text that has been marked by LLM, applying rules and n-gram search to backtick-marked content.

    Args:
        text: Text with LLM-marked symbols (backticks)
        known_symbols: Dictionary of known symbols
        effective_lang: Effective language for processing
        project_root: Project root directory

    Returns:
        Tuple of (matched_symbols, unknown_mentions, lexicon_hits, reconciled_text)
    """
    import re

    from .lexicon import generate_stemmed_aliases, normalize_phrase_with_lexicon

    matched_symbols = []
    unknown_mentions = []
    lexicon_hits = []
    reconciled_text = text

    # Extract backtick-marked phrases for focused processing
    backtick_pattern = r"`([^`]+)`"
    marked_phrases = re.findall(backtick_pattern, text)

    # Apply lexicon normalization to the entire text
    normalized_text, text_lexicon_hits, _ = normalize_phrase_with_lexicon(text, effective_lang, project_root)
    lexicon_hits.extend(text_lexicon_hits)

    # Step 1: Process each marked phrase with direct matching
    for phrase in marked_phrases:
        # Check if phrase directly matches a known symbol
        if phrase in known_symbols:
            if phrase not in matched_symbols:
                matched_symbols.append(phrase)
            continue

        # Apply fuzzy matching to the marked phrase
        best_match = None
        best_score = 0.0

        for symbol_name, symbol_info in known_symbols.items():
            aliases = generate_aliases(symbol_name, symbol_info.get("kind", "unknown"), effective_lang, project_root)
            score = fuzzy_match_symbol(phrase, aliases, threshold=0.7)  # Lower threshold for LLM-marked content

            if score and score > best_score:
                best_score = score
                best_match = symbol_name

        if best_match and best_match not in matched_symbols:
            matched_symbols.append(best_match)
            # Replace the backticked phrase with the matched symbol
            reconciled_text = reconciled_text.replace(f"`{phrase}`", f"`{best_match}`")
        else:
            # Keep the phrase but add to unknown mentions
            if phrase not in unknown_mentions:
                unknown_mentions.append(phrase)

    # Step 2: Generate n-grams ONLY from backticked keywords extracted by RECONCILIATION LLM model
    from .progress import reporter

    reporter.sub_step_with_progress("Processing marked symbols with rules", "extracting n-grams from LLM keywords only", 1, 2)

    # Generate n-grams only from the backticked keywords (LLM extracted)
    ngrams = []

    if marked_phrases:
        # Apply lexicon normalization to marked phrases only
        for phrase in marked_phrases:
            normalized_phrase, _, _ = normalize_phrase_with_lexicon(phrase, effective_lang, project_root)

            # Extract tokens from the marked phrase only
            phrase_tokens = normalize_text(phrase).split()
            normalized_phrase_tokens = normalize_text(normalized_phrase).split()

            # Apply stemming to phrase tokens
            from .lexicon import stem_tokens

            stemmed_phrase = stem_tokens(phrase_tokens, effective_lang)
            stemmed_normalized = stem_tokens(normalized_phrase_tokens, effective_lang)

            # Generate n-grams from the marked phrase tokens only
            ngrams.extend(extract_ngrams(" ".join(phrase_tokens)))
            ngrams.extend(extract_ngrams(" ".join(normalized_phrase_tokens)))
            ngrams.extend(extract_ngrams(" ".join(stemmed_phrase)))
            ngrams.extend(extract_ngrams(" ".join(stemmed_normalized)))

    # Remove duplicates
    ngrams = list(set(ngrams))

    reporter.sub_step_with_progress("Processing marked symbols with rules", f"performing fuzzy matching on {len(ngrams)} keyword n-grams", 2, 2)

    # Initialize debug logger for n-gram comparisons
    from .debug_log import get_debug_logger

    debug_logger = get_debug_logger(project_root)

    # Track improved matches for backticked keywords only
    improved_matches = []

    for ngram in ngrams:
        if len(ngram.strip()) < 2:  # Skip very short ngrams
            continue

        best_match = None
        best_score = 0.0
        ngram_matches = []  # Track all matches for this ngram

        # Only consider symbols that exist in the cache and aren't already matched
        for symbol_name, symbol_info in known_symbols.items():
            if symbol_name in matched_symbols:
                continue  # Skip already matched symbols

            # Generate aliases with stemming support
            aliases = generate_aliases(symbol_name, symbol_info.get("kind", "unknown"), effective_lang, project_root)
            aliases.extend(generate_stemmed_aliases(symbol_name, effective_lang))

            score = fuzzy_match_symbol(ngram, aliases)

            # Record match details for debugging
            match_result = {
                "ngram": ngram,
                "symbol": symbol_name,
                "score": score if score is not None else 0.0,
                "aliases_used": aliases,
                "matched": score is not None and score > 0.8,
            }
            ngram_matches.append(match_result)

            if score and score > best_score:
                best_score = score
                best_match = symbol_name

        # Log n-gram comparison details for debugging
        if ngram_matches:
            debug_logger.log_ngram_comparison("marked_text_ngram_matching", [ngram], "multiple_symbols", [], ngram_matches)

        if best_match and best_match in known_symbols and best_match not in matched_symbols:
            # Map the n-gram back to its source marked phrase for replacement
            for phrase in marked_phrases:
                if ngram.lower() in phrase.lower() or phrase.lower() in ngram.lower():
                    # Replace the original marked phrase with the improved match
                    old_marked = f"`{phrase}`"
                    new_marked = f"`{best_match}`"
                    if old_marked in reconciled_text and new_marked not in reconciled_text:
                        improved_matches.append((old_marked, new_marked))
                        matched_symbols.append(best_match)
                    break
        elif best_score > 0.6:  # Potential match but not in cache
            if ngram not in unknown_mentions:
                unknown_mentions.append(ngram)

    # Apply improved matches, replacing marked phrases with better symbol matches
    for old_marked, new_marked in improved_matches:
        if old_marked in reconciled_text:
            reconciled_text = reconciled_text.replace(old_marked, new_marked)

    return matched_symbols, unknown_mentions, lexicon_hits, reconciled_text


def _process_llm_based(
    text: str, candidate_symbols: List[str], known_symbols: Dict[str, Dict], unresolved_terms: Optional[List[str]] = None
) -> Tuple[List[str], List[str]]:
    """Process text using LLM-based approach."""
    from .progress import reporter

    try:
        from .llm_map import filter_high_confidence_mappings, llm_map_symbols, validate_mappings_against_cache

        # Sub-step 1: Get LLM mappings
        reporter.sub_step_with_progress("Processing with LLM-based matching", f"mapping {len(candidate_symbols)} candidate symbols", 1, 4)
        mappings = llm_map_symbols(text, candidate_symbols)

        # Sub-step 2: Filter high confidence mappings
        reporter.sub_step_with_progress("Processing with LLM-based matching", f"filtering {len(mappings)} mappings by confidence", 2, 4)
        high_conf_mappings = filter_high_confidence_mappings(mappings, min_confidence=0.8)

        # Sub-step 3: Validate against cache
        reporter.sub_step_with_progress("Processing with LLM-based matching", f"validating {len(high_conf_mappings)} high-confidence mappings", 3, 4)
        cache_symbols = set(known_symbols.keys())
        valid_mappings = validate_mappings_against_cache(high_conf_mappings, cache_symbols)

        # Sub-step 4: Extract results
        reporter.sub_step_with_progress("Processing with LLM-based matching", f"extracting {len(valid_mappings)} valid symbols", 4, 4)

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
