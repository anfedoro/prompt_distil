"""
Multilingual lexicon support for identifier reconciliation.

This module provides language-aware normalization of domain terms before fuzzy matching.
Uses small anchor lexicons per language to map source language terms to English anchors,
which are then used in symbol matching. Supports builtin lexicons and per-project overrides.
"""

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

try:
    from snowballstemmer import stemmer as snowball_stemmer
except ImportError:
    snowball_stemmer = None

# Language detection patterns
CYRILLIC_PATTERN = re.compile(r"[\u0400-\u04FF]")
LATIN_PATTERN = re.compile(r"[a-zA-Z]")

# Token normalization patterns
PUNCTUATION_PATTERN = re.compile(r"[^\w\s\-_]")
WHITESPACE_PATTERN = re.compile(r"\s+")

# Supported stemmer languages
STEMMER_LANGUAGES = {"en", "ru", "es"}

# Neutral anchors that don't indicate language preference
NEUTRAL_ANCHORS = {"test", "login", "logging"}


@lru_cache(maxsize=32)
def load_builtin_lexicon(lang: str) -> Dict[str, List[str]]:
    """
    Load JSON from package data: data/lexicons/{lang}.json if present, else {}.

    Args:
        lang: Language code (e.g., 'ru', 'es', 'en')

    Returns:
        Dictionary mapping source language terms to English anchor lists
    """
    try:
        # Get the package directory
        package_dir = Path(__file__).parent.parent
        lexicon_path = package_dir / "data" / "lexicons" / f"{lang}.json"

        if not lexicon_path.exists():
            return {}

        with open(lexicon_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate structure - should be dict with string keys and list values
        if not isinstance(data, dict):
            return {}

        validated = {}
        for key, value in data.items():
            if isinstance(key, str) and isinstance(value, list):
                # Ensure all values in list are strings
                if all(isinstance(v, str) for v in value):
                    validated[key] = value

        return validated

    except Exception:
        # Return empty dict on any error
        return {}


def load_project_lexicon(project_root: Union[str, Path], lang: str) -> Dict[str, List[str]]:
    """
    Load {project_root}/.prompt_distil/lexicon/{lang}.json if present, else {}.

    Args:
        project_root: Root directory of the project
        lang: Language code (e.g., 'ru', 'es', 'en')

    Returns:
        Dictionary mapping source language terms to English anchor lists
    """
    return _load_project_lexicon_cached(str(project_root), lang)


@lru_cache(maxsize=128)
def _load_project_lexicon_cached(project_root_str: str, lang: str) -> Dict[str, List[str]]:
    """Cached implementation of load_project_lexicon."""
    try:
        root_path = Path(project_root_str)
        lexicon_path = root_path / ".prompt_distil" / "lexicon" / f"{lang}.json"

        if not lexicon_path.exists():
            return {}

        with open(lexicon_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Validate structure
        if not isinstance(data, dict):
            return {}

        validated = {}
        for key, value in data.items():
            if isinstance(key, str) and isinstance(value, list):
                if all(isinstance(v, str) for v in value):
                    validated[key] = value

        return validated

    except Exception:
        return {}


def merge_lexicons(builtin: Dict[str, List[str]], override: Dict[str, List[str]]) -> Dict[str, List[str]]:
    """
    Merge builtin and override lexicons. Override wins for conflicts.

    Args:
        builtin: Builtin lexicon dictionary
        override: Project override lexicon dictionary

    Returns:
        Merged dictionary where override values take precedence
    """
    merged = builtin.copy()
    merged.update(override)
    return merged


def detect_lang_fallback(text: str) -> str:
    """
    DEPRECATED: Use get_effective_language instead.
    """
    lang = get_effective_language("auto", text)  # returns str
    return lang


def list_available_lexicons(project_root: Union[str, Path] = ".") -> List[str]:
    """Get list of available lexicon languages."""
    available = []
    for lang in ["en", "ru", "es"]:
        builtin = load_builtin_lexicon(lang)
        project = load_project_lexicon(project_root, lang) if project_root else {}
        if builtin or project:
            available.append(lang)
    return available


def detect_language(text: str, asr_language: str = "auto", available_lexicons: List[str] = None) -> Tuple[str, Dict]:
    """
    Detect language with voting and metadata.

    Returns:
        Tuple of (language_code, metadata_dict)
    """
    if asr_language and asr_language != "auto":
        return asr_language, {"reason": "asr", "votes": {}}

    if not available_lexicons:
        available_lexicons = ["en", "ru", "es"]

    from collections import Counter

    votes = Counter()

    # Count lexicon hits for each language
    tokens = tokenize_normalize(text)
    for lang in available_lexicons:
        builtin_lex = load_builtin_lexicon(lang)
        if not builtin_lex:
            continue

        for token in tokens:
            if token in builtin_lex:
                # Check if any anchors are neutral
                anchors = builtin_lex[token]
                if not any(anchor in NEUTRAL_ANCHORS for anchor in anchors):
                    votes[lang] += 1

    vote_counts = dict(votes)

    if not votes:
        fallback = detect_lang_fallback_simple(text)
        return fallback, {"reason": "script", "votes": vote_counts}

    # Get top language
    most_common = votes.most_common(1)
    top_lang = most_common[0][0] if most_common else "en"

    return top_lang, {"reason": "lexicon", "votes": vote_counts}


def detect_lang_fallback_simple(text: str) -> str:
    """Simple script-based language detection."""
    if not text:
        return "en"

    # Count character types
    cyrillic_count = len(CYRILLIC_PATTERN.findall(text))
    latin_count = len(LATIN_PATTERN.findall(text))

    # If significant Cyrillic presence, assume Russian
    if cyrillic_count > 0 and cyrillic_count >= latin_count * 0.3:
        return "ru"

    # Check for Spanish-specific patterns
    spanish_indicators = ["ñ", "á", "é", "í", "ó", "ú", "ü", "¿", "¡"]
    if any(indicator in text.lower() for indicator in spanish_indicators):
        return "es"

    # Default to English
    return "en"


@lru_cache(maxsize=16)
def get_stemmer(lang: str):
    """
    Get a SnowballStemmer for the specified language.

    Args:
        lang: Language code (e.g., 'ru', 'es', 'en')

    Returns:
        SnowballStemmer instance if available and supported, None otherwise
    """
    if not snowball_stemmer or lang not in STEMMER_LANGUAGES:
        return None

    try:
        if lang == "ru":
            return snowball_stemmer("russian")
        elif lang == "es":
            return snowball_stemmer("spanish")
        elif lang == "en":
            return snowball_stemmer("english")
        else:
            return None
    except Exception:
        return None


@lru_cache(maxsize=1024)
def stem_token(token: str, lang: str) -> str:
    """
    Stem a single token using the appropriate stemmer.

    Args:
        token: Token to stem
        lang: Language code

    Returns:
        Stemmed token, or original token if stemming not available
    """
    stemmer = get_stemmer(lang)
    if not stemmer or not token:
        return token

    try:
        # snowballstemmer has stemWord method
        return stemmer.stemWord(token)  # type: ignore
    except Exception:
        return token


def stem_tokens(tokens: List[str], lang: str) -> List[str]:
    """
    Stem a list of tokens using the appropriate stemmer.

    Args:
        tokens: List of tokens to stem
        lang: Language code

    Returns:
        List of stemmed tokens
    """
    stemmer = get_stemmer(lang)
    if not stemmer:
        return tokens

    def tokenize_normalize_stem(text: str, lang: str = "en") -> Tuple[List[str], List[str]]:
        """
        Tokenize, normalize, and stem text.

        Args:
            text: Input text to process
            lang: Language code for stemming

        Returns:
            Tuple of (original_tokens, stemmed_tokens)
        """
        tokens = tokenize_normalize(text)
        stemmed = stem_tokens(tokens, lang)
        return tokens, stemmed

    stemmed = []
    for token in tokens:
        try:
            if token:
                stemmed.append(stemmer.stemWord(token))  # type: ignore
            else:
                stemmed.append(token)
        except Exception:
            stemmed.append(token)

    return stemmed


def tokenize_normalize(text: str) -> List[str]:
    """
    Lowercase, strip punctuation, collapse spaces; return tokens.

    Args:
        text: Input text to tokenize and normalize

    Returns:
        List of normalized tokens
    """
    if not text:
        return []

    # Lowercase
    normalized = text.lower()

    # Remove punctuation except underscores and hyphens
    normalized = PUNCTUATION_PATTERN.sub(" ", normalized)

    # Collapse whitespace
    normalized = WHITESPACE_PATTERN.sub(" ", normalized)

    # Split into tokens and filter empty
    tokens = [token.strip() for token in normalized.split() if token.strip()]

    return tokens


def apply_lexicon_tokens(tokens: List[str], lex: Dict[str, List[str]], lang: str = "en") -> Tuple[List[str], List[str], Dict[str, str]]:
    """
    Replace tokens via lex (keys in source lang -> first EN anchor) with stemming support.

    Args:
        tokens: List of tokens to process
        lex: Lexicon mapping source terms to English anchors
        lang: Language code for stemming

    Returns:
        Tuple of (replaced_tokens, hits_list, stem_map) where:
        - replaced_tokens: tokens with lexicon replacements
        - hits_list: original source terms that were replaced
        - stem_map: mapping of stemmed forms to original lexicon keys
    """
    if not tokens or not lex:
        return tokens.copy(), [], {}

    # Build stemmed lexicon for fuzzy matching
    stem_map = {}
    stemmed_lex = {}

    for lex_key, anchors in lex.items():
        stemmed_key = stem_token(lex_key, lang)
        stem_map[stemmed_key] = lex_key
        stemmed_lex[stemmed_key] = anchors

    replaced_tokens = []
    hits_list = []

    for token in tokens:
        # Try exact match first
        if token in lex and lex[token]:
            anchor = lex[token][0]
            replaced_tokens.append(anchor)
            hits_list.append(token)
        else:
            # Try stemmed match
            stemmed_token = stem_token(token, lang)
            if stemmed_token in stemmed_lex and stemmed_lex[stemmed_token]:
                anchor = stemmed_lex[stemmed_token][0]
                replaced_tokens.append(anchor)
                # Record the original lexicon key that matched
                original_key = stem_map[stemmed_token]
                hits_list.append(original_key)
            else:
                replaced_tokens.append(token)

    return replaced_tokens, hits_list, stem_map


def normalize_phrase_with_lexicon(text: str, lang: str, project_root: Union[str, Path]) -> Tuple[str, List[str], Dict[str, str]]:
    """
    Normalize a phrase using the appropriate lexicon for the language.

    Args:
        text: Input text to normalize
        lang: Language code
        project_root: Project root for loading override lexicon

    Returns:
        Tuple of (normalized_text, lexicon_hits, stem_map)
    """
    # Load and merge lexicons
    builtin_lex = load_builtin_lexicon(lang)
    project_lex = load_project_lexicon(project_root, lang)
    merged_lex = merge_lexicons(builtin_lex, project_lex)

    # Tokenize and apply lexicon with stemming
    tokens = tokenize_normalize(text)
    replaced_tokens, hits, stem_map = apply_lexicon_tokens(tokens, merged_lex, lang)

    # Recompose normalized text
    normalized_text = " ".join(replaced_tokens)

    return normalized_text, hits, stem_map


def generate_lexicon_aware_aliases(phrase: str, lang: str, project_root: Union[str, Path]) -> List[str]:
    """
    Generate aliases for a phrase using lexicon normalization.

    Args:
        phrase: Input phrase to generate aliases for
        lang: Language code
        project_root: Project root for loading override lexicon

    Returns:
        List of alias variations
    """
    return _generate_lexicon_aware_aliases_cached(phrase, lang, str(project_root))


@lru_cache(maxsize=512)
def _generate_lexicon_aware_aliases_cached(phrase: str, lang: str, project_root_str: str) -> List[str]:
    """Cached implementation of generate_lexicon_aware_aliases."""
    aliases = [phrase]  # Include original

    # Get normalized version
    normalized, _, _ = normalize_phrase_with_lexicon(phrase, lang, project_root_str)
    if normalized != phrase:
        aliases.append(normalized)

    # Generate variations of normalized text
    if normalized:
        # Snake case version
        snake_case = normalized.replace(" ", "_")
        aliases.append(snake_case)

        # Hyphenated version
        hyphenated = normalized.replace(" ", "-")
        aliases.append(hyphenated)

        # Camel case versions
        words = normalized.split()
        if len(words) > 1:
            # PascalCase
            pascal_case = "".join(word.capitalize() for word in words)
            aliases.append(pascal_case)

            # camelCase
            camel_case = words[0] + "".join(word.capitalize() for word in words[1:])
            aliases.append(camel_case)

    # Remove duplicates while preserving order
    seen = set()
    unique_aliases = []
    for alias in aliases:
        if alias and alias not in seen:
            seen.add(alias)
            unique_aliases.append(alias)

    return unique_aliases


@lru_cache(maxsize=512)
def generate_stemmed_aliases(phrase: str, lang: str) -> List[str]:
    """
    Generate stemmed variations of a phrase for improved matching.

    Args:
        phrase: Input phrase to generate stemmed aliases for
        lang: Language code for stemming

    Returns:
        List of stemmed alias variations
    """
    aliases = [phrase]

    # Get stemmed version
    tokens = tokenize_normalize(phrase)
    stemmed_tokens = stem_tokens(tokens, lang)

    if stemmed_tokens != tokens:
        # Add stemmed variations
        stemmed_phrase = " ".join(stemmed_tokens)
        aliases.append(stemmed_phrase)

        # Add stemmed snake_case and other variations
        stemmed_snake = stemmed_phrase.replace(" ", "_")
        stemmed_hyphen = stemmed_phrase.replace(" ", "-")
        aliases.extend([stemmed_snake, stemmed_hyphen])

        # Add camel case versions of stemmed text
        if len(stemmed_tokens) > 1:
            pascal_case = "".join(word.capitalize() for word in stemmed_tokens)
            camel_case = stemmed_tokens[0] + "".join(word.capitalize() for word in stemmed_tokens[1:])
            aliases.extend([pascal_case, camel_case])

    # Remove duplicates while preserving order
    seen = set()
    unique_aliases = []
    for alias in aliases:
        if alias and alias not in seen:
            seen.add(alias)
            unique_aliases.append(alias)

    return unique_aliases


def get_effective_language(
    asr_language: str,
    text: str,
    project_root: Union[str, Path] = None,
    *,
    return_meta: bool = False,
):
    """
    Back-compatible helper for language detection.

    - If called with 2 args: return lang:str
    - If called with 3rd arg project_root OR return_meta=True: return (lang:str, meta:dict)

    Args:
        asr_language: Language detected by ASR (may be "auto" or None)
        text: Text to analyze if ASR language is not available
        project_root: Project root for loading lexicons (optional)
        return_meta: Whether to return metadata

    Returns:
        Language code (str) or tuple (lang:str, meta:dict)
    """
    # Determine if we should return metadata
    should_return_meta = project_root is not None or return_meta

    # Get available lexicons
    available = list_available_lexicons(project_root) if project_root else ["en", "ru", "es"]

    # Detect language with metadata
    lang, meta = detect_language(text, asr_language, available)

    # Return based on call signature
    if should_return_meta:
        return lang, meta
    else:
        return lang
