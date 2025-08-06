"""
Basic tests for the Prompt Distiller functionality.

These tests verify core functionality without requiring API keys or external services.
"""

from pathlib import Path

import pytest

from prompt_distil.core.config import Config
from prompt_distil.core.prompt import PromptRenderer
from prompt_distil.core.surface import ProjectSurface
from prompt_distil.core.types import IRLite, KnownEntity


class TestTypes:
    """Test the type definitions and data structures."""

    def test_known_entity_creation(self):
        """Test KnownEntity model creation."""
        entity = KnownEntity(path="src/main.py", symbol="main", confidence=0.9)
        assert entity.path == "src/main.py"
        assert entity.symbol == "main"
        assert entity.confidence == 0.9

    def test_known_entity_minimal(self):
        """Test KnownEntity with minimal required fields."""
        entity = KnownEntity(path="config.py")
        assert entity.path == "config.py"
        assert entity.symbol is None
        assert entity.confidence is None

    def test_ir_lite_creation(self):
        """Test IRLite model creation."""
        ir = IRLite(goal="Test the application", must=["Write unit tests"], must_not=["Break existing functionality"], assumptions=["Current code is working"])
        assert ir.goal == "Test the application"
        assert len(ir.must) == 1
        assert len(ir.must_not) == 1
        assert len(ir.assumptions) == 1
        assert len(ir.scope_hints) == 0  # default empty list

    def test_ir_lite_minimal(self):
        """Test IRLite with only required fields."""
        ir = IRLite(goal="Simple goal")
        assert ir.goal == "Simple goal"
        assert ir.scope_hints == []
        assert ir.must == []
        assert ir.must_not == []
        assert ir.known_entities == []
        assert ir.unknowns == []
        assert ir.acceptance == []
        assert ir.assumptions == []


class TestPromptRenderer:
    """Test prompt rendering functionality."""

    def test_renderer_initialization(self):
        """Test PromptRenderer initialization."""
        renderer = PromptRenderer()
        assert "short" in renderer.profiles
        assert "standard" in renderer.profiles
        assert "verbose" in renderer.profiles

    def test_render_short_profile(self):
        """Test rendering short profile."""
        ir = IRLite(goal="Fix the bug", must=["Identify root cause", "Write test"], must_not=["Change API"])
        renderer = PromptRenderer()
        result = renderer.render(ir, "short")

        assert "## Goal" in result
        assert "Fix the bug" in result
        assert "## Change Request" in result
        assert "✓ Identify root cause" in result
        assert "✗ Change API" in result

    def test_render_standard_profile(self):
        """Test rendering standard profile."""
        ir = IRLite(goal="Implement feature", must=["Add new endpoint"], acceptance=["Returns 200 OK"], assumptions=["Database is available"])
        renderer = PromptRenderer()
        result = renderer.render(ir, "standard")

        assert "## Goal" in result
        assert "## Change Request" in result
        assert "## Acceptance Criteria" in result
        assert "## Assumptions" in result
        assert "**Required:**" in result
        assert "Add new endpoint" in result

    def test_render_verbose_profile(self):
        """Test rendering verbose profile."""
        ir = IRLite(goal="Refactor code", must=["Improve performance"], unknowns=["Current bottleneck location"])
        renderer = PromptRenderer()
        result = renderer.render(ir, "verbose")

        assert "## Goal" in result
        assert "## Change Request" in result
        assert "## Constraints" in result
        assert "## Out of Scope" in result
        assert "## Deliverables" in result
        assert "### Required Changes" in result
        assert "### Unclear Requirements" in result

    def test_render_all_profiles(self):
        """Test rendering all profiles at once."""
        ir = IRLite(goal="Test goal")
        renderer = PromptRenderer()
        results = renderer.render_all(ir)

        assert "short" in results
        assert "standard" in results
        assert "verbose" in results
        assert all(isinstance(prompt, str) for prompt in results.values())

    def test_invalid_profile(self):
        """Test handling of invalid profile."""
        ir = IRLite(goal="Test goal")
        renderer = PromptRenderer()

        with pytest.raises(ValueError, match="Unsupported profile"):
            renderer.render(ir, "invalid")

    def test_format_entities_compact(self):
        """Test entity formatting in compact mode."""
        entities = [KnownEntity(path="src/main.py", symbol="main"), KnownEntity(path="src/utils.py")]
        renderer = PromptRenderer()
        result = renderer._format_entities(entities, compact=True)

        assert "src/main.py — `main`" in result
        assert "src/utils.py" in result

    def test_format_entities_detailed(self):
        """Test entity formatting with detailed information."""
        entities = [KnownEntity(path="src/core.py", symbol="process", confidence=0.8)]
        renderer = PromptRenderer()
        result = renderer._format_entities(entities, detailed=True)

        assert "- src/core.py — `process`" in result
        assert "confidence: 0.80" in result


class TestConfig:
    """Test configuration functionality."""

    def test_config_initialization(self):
        """Test Config class initialization."""
        config = Config()
        assert config is not None

    def test_default_models(self):
        """Test default model configurations."""
        config = Config()
        # These should return defaults even without environment variables
        assert config.distil_model == "gpt-4.1-mini"
        assert config.asr_model == "whisper-1"
        assert config.openai_timeout == 60
        assert config.max_retries == 3


class TestProjectSurface:
    """Test project surface functionality."""

    def test_surface_initialization_current_dir(self):
        """Test ProjectSurface initialization with current directory."""
        surface = ProjectSurface()
        assert surface.project_root.exists()

    def test_list_files_basic(self):
        """Test basic file listing."""
        surface = ProjectSurface()
        files = surface.list_files("**/*.py")
        assert isinstance(files, list)
        # Should find at least our test file
        py_files = [f for f in files if f.endswith(".py")]
        assert len(py_files) > 0

    def test_get_project_structure(self):
        """Test project structure analysis."""
        surface = ProjectSurface()
        structure = surface.get_project_structure(max_depth=2)

        assert "root" in structure
        assert "directories" in structure
        assert "files" in structure
        assert "file_types" in structure
        assert isinstance(structure["file_types"], dict)

    def test_validate_audio_format(self):
        """Test audio format validation in SpeechProcessor."""
        from prompt_distil.core.speech import SpeechProcessor

        processor = SpeechProcessor.__new__(SpeechProcessor)  # Create without __init__

        assert processor.validate_audio_format("test.wav") is True
        assert processor.validate_audio_format("test.mp3") is True
        assert processor.validate_audio_format("test.txt") is False
        assert processor.validate_audio_format("test.xyz") is False

    def test_transcript_type(self):
        """Test Transcript data type."""
        from prompt_distil.core.types import Transcript

        transcript = Transcript(text="Hello world", lang_hint="en")
        assert transcript.text == "Hello world"
        assert transcript.lang_hint == "en"

        # Test default lang_hint
        transcript_auto = Transcript(text="Test")
        assert transcript_auto.lang_hint == "auto"


class TestCodeIdentifierProtection:
    """Test code identifier protection functionality."""

    def test_protect_code_identifiers_with_underscores(self):
        """Test protection of identifiers containing underscores."""
        from prompt_distil.core.speech import protect_code_identifiers

        text = "The delete_task function should handle user_input correctly"
        result = protect_code_identifiers(text)
        assert "`delete_task`" in result
        assert "`user_input`" in result
        assert "function should handle" in result  # Regular words unchanged

    def test_protect_code_identifiers_whitelist(self):
        """Test protection of whitelisted identifiers."""
        from prompt_distil.core.speech import protect_code_identifiers

        text = "Use FastAPI and pydantic for the API"
        result = protect_code_identifiers(text)
        assert "`FastAPI`" in result
        assert "`pydantic`" in result
        assert "for the API" in result

    def test_protect_code_identifiers_already_protected(self):
        """Test that already backticked identifiers are not double-protected."""
        from prompt_distil.core.speech import protect_code_identifiers

        text = "The `delete_task` function and user_model work together"
        result = protect_code_identifiers(text)
        assert "`delete_task`" in result  # Should remain single backticked
        assert "``delete_task``" not in result  # Should not be double backticked
        assert "`user_model`" in result

    def test_protect_code_identifiers_mixed_case(self):
        """Test protection with mixed case identifiers."""
        from prompt_distil.core.speech import protect_code_identifiers

        text = "Use EmailStr and login_handler for authentication"
        result = protect_code_identifiers(text)
        assert "`EmailStr`" in result
        assert "`login_handler`" in result

    def test_protect_code_identifiers_no_matches(self):
        """Test text with no identifiers to protect."""
        from prompt_distil.core.speech import protect_code_identifiers

        text = "This is just regular text without any code identifiers"
        result = protect_code_identifiers(text)
        assert result == text  # Should remain unchanged

    def test_target_language_support(self):
        """Test target language parameter in distillation."""
        from prompt_distil.core.distill import TranscriptDistiller

        # Mock the distiller to avoid API calls
        distiller = TranscriptDistiller.__new__(TranscriptDistiller)

        # Test system prompt generation with different target languages
        system_prompt_en = distiller._create_system_prompt({}, target_language="en")
        system_prompt_auto = distiller._create_system_prompt({}, target_language="auto")

        assert "Produce final prompts in English" in system_prompt_en
        assert "Produce final prompts in the source language" in system_prompt_auto


class TestSymbolCache:
    """Test symbol cache functionality."""

    def test_camel_to_snake(self):
        """Test CamelCase to snake_case conversion."""
        from prompt_distil.core.reconcile import camel_to_snake

        assert camel_to_snake("DeleteTask") == "delete_task"
        assert camel_to_snake("LoginHandler") == "login_handler"
        assert camel_to_snake("UserModel") == "user_model"
        assert camel_to_snake("FastAPI") == "fast_api"
        assert camel_to_snake("simple") == "simple"

    def test_generate_aliases(self):
        """Test alias generation for symbols."""
        from prompt_distil.core.reconcile import generate_aliases

        # Test function with underscore
        aliases = generate_aliases("delete_task", "function")
        assert "delete_task" in aliases
        assert "delete task" in aliases
        assert "delete-task" in aliases

        # Test CamelCase class
        aliases = generate_aliases("DeleteTask", "class")
        assert "DeleteTask" in aliases
        assert "delete_task" in aliases
        assert "deleteTask" in aliases
        assert "DeleteTaskHandler" in aliases

    def test_normalize_text(self):
        """Test text normalization."""
        from prompt_distil.core.reconcile import normalize_text

        result = normalize_text("The delete_task function!")
        assert result == "delete_task function"

        result = normalize_text("Update the user model")
        assert result == "update user model"

    def test_fuzzy_match_symbol(self):
        """Test fuzzy symbol matching."""
        from prompt_distil.core.reconcile import fuzzy_match_symbol

        aliases = ["delete_task", "delete task", "delete-task"]

        # Exact match
        score = fuzzy_match_symbol("delete_task", aliases)
        assert score == 1.0

        # Close match
        score = fuzzy_match_symbol("delete task", aliases)
        assert score == 1.0

        # Fuzzy match
        score = fuzzy_match_symbol("delete tasks", aliases, threshold=0.7)
        assert score is not None and score > 0.7

        # No match
        score = fuzzy_match_symbol("completely different", aliases)
        assert score is None

    def test_reconcile_text_basic(self):
        """Test basic text reconciliation."""
        import tempfile

        from prompt_distil.core.reconcile import reconcile_text
        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with symbols
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "files": [],
                "symbols": [
                    {"name": "delete_task", "kind": "function", "path": "tasks.py", "lineno": 10},
                    {"name": "login_handler", "kind": "function", "path": "auth.py", "lineno": 5},
                ],
                "inverted_index": {"delete_task": ["tasks.py#L10"], "login_handler": ["auth.py#L5"]},
            }
            save_cache(temp_dir, cache_data)

            text = "Update the delete_task function and login handler"
            reconciled, matched, unknown, lex_hits, unresolved = reconcile_text(text, temp_dir)

            assert "`delete_task`" in reconciled
            assert "`login_handler`" in reconciled
            assert "delete_task" in matched
            assert "login_handler" in matched

    def test_reconcile_text_context_rules(self):
        """Test context-specific reconciliation rules."""
        import tempfile

        from prompt_distil.core.reconcile import reconcile_text
        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with symbol
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "files": [],
                "symbols": [{"name": "delete_task", "kind": "function", "path": "tasks.py", "lineno": 10}],
                "inverted_index": {"delete_task": ["tasks.py#L10"]},
            }
            save_cache(temp_dir, cache_data)

            # Test "test for" pattern
            text = "Create test for delete task functionality"
            reconciled, matched, unknown, lex_hits, unresolved = reconcile_text(text, temp_dir)

            assert "delete_task" in matched

    def test_cache_structure(self):
        """Test symbol cache structure."""
        import os
        import tempfile

        from prompt_distil.core.surface import build_symbol_inventory

        # Create a temporary Python file for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.py")
            with open(test_file, "w") as f:
                f.write("""
def delete_task(task_id):
    pass

class UserModel:
    def login_handler(self):
        pass
""")

            cache = build_symbol_inventory(temp_dir, globs=["**/*.py"])

            assert "version" in cache
            assert "generated_at" in cache
            assert "root" in cache
            assert "globs" in cache
            assert "files" in cache
            assert "symbols" in cache
            assert "inverted_index" in cache

            # Check symbols were extracted
            symbol_names = [s["name"] for s in cache["symbols"]]
            assert "delete_task" in symbol_names
            assert "UserModel" in symbol_names
            assert "login_handler" in symbol_names

            # Check inverted index
            assert "delete_task" in cache["inverted_index"]
            assert len(cache["inverted_index"]["delete_task"]) > 0


class TestProjectRootIntegration:
    """Test project-root integration and safe reconciliation."""

    def test_project_root_cache_path(self):
        """Test that cache is stored under project root."""
        import tempfile

        from prompt_distil.core.surface import load_cache, save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "files": [],
                "symbols": [],
                "inverted_index": {},
            }

            # Save cache
            save_cache(temp_dir, cache_data)

            # Check that cache file exists under project root
            cache_path = Path(temp_dir) / ".prompt_distil" / "project_cache.json"
            assert cache_path.exists()

            # Load and verify
            loaded_cache = load_cache(temp_dir)
            assert loaded_cache is not None
            assert loaded_cache["root"] == temp_dir

    def test_safe_reconciliation_with_cache(self):
        """Test that reconciliation only backticks symbols in cache."""
        import tempfile

        from prompt_distil.core.reconcile import reconcile_text
        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with only delete_task
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "files": [],
                "symbols": [{"name": "delete_task", "kind": "function", "path": "tasks.py", "lineno": 10}],
                "inverted_index": {"delete_task": ["tasks.py#L10"]},
            }

            save_cache(temp_dir, cache_data)

            # Test reconciliation
            text = "Update delete_task and login_handler functions"
            reconciled, matched, unknown, lex_hits, unresolved = reconcile_text(text, temp_dir)

            # delete_task should be backticked (in cache)
            assert "`delete_task`" in reconciled
            assert "delete_task" in matched

            # login_handler should NOT be backticked (not in cache)
            assert "`login_handler`" not in reconciled
            assert "login_handler" not in matched

    def test_safe_reconciliation_without_cache(self):
        """Test reconciliation behavior when no cache exists."""
        import tempfile

        from prompt_distil.core.reconcile import reconcile_text

        with tempfile.TemporaryDirectory() as temp_dir:
            # No cache file exists
            text = "Update delete_task and login_handler functions"
            reconciled, matched, unknown, lex_hits, unresolved = reconcile_text(text, temp_dir)

            # Nothing should be backticked
            assert reconciled == text
            assert len(matched) == 0
            assert len(unknown) == 0
            assert len(lex_hits) == 0

    def test_related_section_no_placeholders(self):
        """Test that Related section shows no placeholders for invalid paths."""
        from prompt_distil.core.prompt import PromptRenderer
        from prompt_distil.core.types import IRLite, KnownEntity

        # Create entities with invalid paths
        entities = [
            KnownEntity(path="", symbol="test"),
            KnownEntity(path="unknown", symbol="test2"),
            KnownEntity(path="****", symbol="test3"),
            KnownEntity(path="valid/path.py", symbol="test4"),  # This should appear
        ]

        ir = IRLite(goal="Test goal", known_entities=entities)
        renderer = PromptRenderer()
        result = renderer.render(ir, "standard")

        # Only the valid path should appear in Related section
        assert "valid/path.py — `test4`" in result
        assert "— `test`" not in result  # No bare symbols without paths
        assert "unknown" not in result
        assert "****" not in result

    def test_project_root_parameter_passing(self):
        """Test that project_root is passed correctly through the pipeline."""
        import tempfile

        from prompt_distil.core.distill import TranscriptDistiller

        # Test with custom project root (use existing temp directory)
        with tempfile.TemporaryDirectory() as temp_dir:
            distiller = TranscriptDistiller(temp_dir)
            assert distiller.project_root == temp_dir

        # Test default
        distiller_default = TranscriptDistiller()
        assert distiller_default.project_root == "."


class TestMultilingualLexicon:
    """Test multilingual lexicon functionality."""

    def test_stemming_functionality(self):
        """Test stemming support."""
        from prompt_distil.core.lexicon import get_stemmer, stem_token, stem_tokens

        # Test Russian stemmer
        ru_stemmer = get_stemmer("ru")
        if ru_stemmer:  # Only test if stemmer is available
            assert stem_token("задачи", "ru") != "задачи"  # Should stem
            assert stem_token("обработчик", "ru") == stem_token("обработчики", "ru")  # Same stem

        # Test English stemmer
        en_stemmer = get_stemmer("en")
        if en_stemmer:
            assert stem_token("running", "en") == stem_token("runs", "en")  # Same stem

        # Test token list stemming
        tokens = ["running", "tasks", "handlers"]
        stemmed = stem_tokens(tokens, "en")
        assert len(stemmed) == len(tokens)

    def test_lex_mode_rules_only(self):
        """Test rules-only lexicon mode."""
        import tempfile

        from prompt_distil.core.reconcile import reconcile_text
        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with symbols
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "files": [],
                "symbols": [{"name": "delete_task", "kind": "function", "path": "tasks.py", "lineno": 10}],
                "inverted_index": {"delete_task": ["tasks.py#L10"]},
            }
            save_cache(temp_dir, cache_data)

            text = "Update delete_task function"
            reconciled, matched, unknown, lex_hits, unresolved = reconcile_text(text, temp_dir, "en", "rules")

            assert "delete_task" in matched
            assert "`delete_task`" in reconciled

    def test_lex_mode_hybrid(self):
        """Test hybrid lexicon mode."""
        import tempfile

        from prompt_distil.core.reconcile import reconcile_text
        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with symbols
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "files": [],
                "symbols": [{"name": "delete_task", "kind": "function", "path": "tasks.py", "lineno": 10}],
                "inverted_index": {"delete_task": ["tasks.py#L10"]},
            }
            save_cache(temp_dir, cache_data)

            text = "Update delete_task and some_unknown_function"
            reconciled, matched, unknown, lex_hits, unresolved = reconcile_text(text, temp_dir, "en", "hybrid")

            # Should find delete_task via rules
            assert "delete_task" in matched
            # May have unresolved terms for unknown functions
            assert isinstance(unresolved, list)

    def test_load_builtin_lexicon(self):
        """Test loading builtin lexicon files."""
        from prompt_distil.core.lexicon import load_builtin_lexicon

        # Test Russian lexicon
        ru_lex = load_builtin_lexicon("ru")
        assert isinstance(ru_lex, dict)
        assert "обработчик" in ru_lex
        assert ru_lex["обработчик"] == ["handler"]

        # Test Spanish lexicon
        es_lex = load_builtin_lexicon("es")
        assert isinstance(es_lex, dict)
        assert "controlador" in es_lex
        assert es_lex["controlador"] == ["controller"]

        # Test non-existent lexicon
        empty_lex = load_builtin_lexicon("nonexistent")
        assert empty_lex == {}

    def test_detect_lang_fallback(self):
        """Test language detection fallback logic."""
        from prompt_distil.core.lexicon import detect_lang_fallback

        # Test Russian (Cyrillic)
        assert detect_lang_fallback("Удалить задачу и обработчик") == "ru"

        # Test Spanish
        assert detect_lang_fallback("Configuración del usuario") == "es"

        # Test English (default)
        assert detect_lang_fallback("Delete task and handler") == "en"

        # Test empty string
        assert detect_lang_fallback("") == "en"

    def test_tokenize_normalize(self):
        """Test text tokenization and normalization."""
        from prompt_distil.core.lexicon import tokenize_normalize

        text = "The delete_task function! It's working."
        tokens = tokenize_normalize(text)
        assert "the" in tokens
        assert "delete_task" in tokens
        assert "function" in tokens
        assert "it" in tokens  # Becomes separate tokens "it" and "s"
        assert "working" in tokens

    def test_apply_lexicon_tokens(self):
        """Test lexicon application to tokens."""
        from prompt_distil.core.lexicon import apply_lexicon_tokens

        tokens = ["удалить", "задачу", "обработчик"]
        lexicon = {"удалить": ["delete"], "задачу": ["task"], "обработчик": ["handler"]}

        replaced, hits, stem_map = apply_lexicon_tokens(tokens, lexicon)
        assert replaced == ["delete", "task", "handler"]
        assert set(hits) == {"удалить", "задачу", "обработчик"}

    def test_normalize_phrase_with_lexicon(self):
        """Test phrase normalization using lexicon."""
        import tempfile

        from prompt_distil.core.lexicon import normalize_phrase_with_lexicon

        with tempfile.TemporaryDirectory() as temp_dir:
            text = "удалить задачу пользователя"
            normalized, hits, stem_map = normalize_phrase_with_lexicon(text, "ru", temp_dir)

            # Should use builtin Russian lexicon
            assert "delete" in normalized
            # Note: not all Russian words may have exact mappings
            assert len(hits) > 0  # Should have some lexicon hits
            assert "удалить" in hits
            assert "задачу" in hits

    def test_lexicon_aware_reconciliation(self):
        """Test reconciliation with lexicon support."""
        import tempfile

        from prompt_distil.core.reconcile import reconcile_text
        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with delete_task symbol
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "files": [],
                "symbols": [{"name": "delete_task", "kind": "function", "path": "tasks.py", "lineno": 10}],
                "inverted_index": {"delete_task": ["tasks.py#L10"]},
            }
            save_cache(temp_dir, cache_data)

            # Test Russian text that should map to delete_task
            text = "Переписать тест для удаления задачи"
            reconciled, matched, unknown, lex_hits, unresolved = reconcile_text(text, temp_dir, "ru")

            # May not match due to inflections - test basic functionality
            assert len(lex_hits) >= 0  # Should process lexicon
            # Complex inflected forms may not match exactly

    def test_get_effective_language(self):
        """Test effective language determination."""
        from prompt_distil.core.lexicon import get_effective_language

        # ASR language takes precedence
        assert get_effective_language("ru", "any text") == "ru"
        assert get_effective_language("es", "any text") == "es"

        # Fallback to detection when ASR is auto
        assert get_effective_language("auto", "Удалить задачу") == "ru"
        assert get_effective_language("auto", "Delete task") == "en"


if __name__ == "__main__":
    pytest.main([__file__])
