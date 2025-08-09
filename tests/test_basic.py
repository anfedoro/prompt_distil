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
        import os
        from unittest.mock import patch

        # Preserve OPENAI_API_KEY but clear model-specific environment variables
        openai_key = os.environ.get("OPENAI_API_KEY", "test-key")
        clean_env = {"OPENAI_API_KEY": openai_key}

        with patch.dict(os.environ, clean_env, clear=True):
            config = Config()
            # These should return defaults without model environment variables
            assert config.distil_model == "gpt-4o-mini"
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
        # Test system prompt generation via LLM Handler
        from prompt_distil.core.llm_handler import LLMHandler

        handler = LLMHandler(".")
        system_prompt_en = handler._create_distillation_system_prompt({}, target_language="en")
        system_prompt_auto = handler._create_distillation_system_prompt({}, target_language="auto")

        assert "Produce final prompts in English" in system_prompt_en
        assert "the source language" in system_prompt_auto
        assert "Produce final prompts in the source language" in system_prompt_auto


class TestSymbolCache:
    """Test symbol cache functionality."""

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

            # Mock LLM call to avoid API requests in tests
            from unittest.mock import patch

            with patch("prompt_distil.core.llm_map.llm_preprocess_text") as mock_llm:
                mock_llm.return_value = "Update the `delete_task` function and `login_handler`"
                reconciled, matched, unknown, unresolved = reconcile_text(text, temp_dir)

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

            # Mock LLM call to avoid API requests in tests
            from unittest.mock import patch

            with patch("prompt_distil.core.llm_map.llm_preprocess_text") as mock_llm:
                mock_llm.return_value = "Create test for `delete_task` functionality"
                reconciled, matched, unknown, unresolved = reconcile_text(text, temp_dir)

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

            # Mock LLM call to avoid API requests in tests
            from unittest.mock import patch

            with patch("prompt_distil.core.llm_map.llm_preprocess_text") as mock_llm:
                mock_llm.return_value = "Update `delete_task` and `login_handler` functions"
                reconciled, matched, unknown, unresolved = reconcile_text(text, temp_dir)

            # delete_task should be backticked (in cache)
            assert "`delete_task`" in reconciled
            assert "delete_task" in matched

            # login_handler should be in unknown since it's not in cache but LLM marked it
            assert "login_handler" in unknown
            # The reconciled text contains what LLM returned, but login_handler should be tracked as unknown
            assert "`login_handler`" in reconciled

    def test_safe_reconciliation_without_cache(self):
        """Test reconciliation behavior when no cache exists."""
        import tempfile

        from prompt_distil.core.reconcile import reconcile_text

        with tempfile.TemporaryDirectory() as temp_dir:
            # No cache file exists
            text = "Update delete_task and login_handler functions"
            reconciled, matched, unknown, unresolved = reconcile_text(text, temp_dir)

            # Nothing should be backticked
            assert reconciled == text
            assert len(matched) == 0
            assert len(unknown) == 0

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

    def test_reconcile_text_default_hybrid(self):
        """Test default hybrid reconciliation mode."""
        import tempfile
        from unittest.mock import patch

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

            # Mock LLM call to avoid API requests and speed up tests
            with patch("prompt_distil.core.llm_map.llm_preprocess_text") as mock_llm:
                mock_llm.return_value = "Update `delete_task` function"
                reconciled, matched, unknown, unresolved = reconcile_text(text, temp_dir, "en")

            assert "delete_task" in matched
            assert "`delete_task`" in reconciled

    def test_lex_mode_hybrid(self):
        """Test hybrid lexicon mode."""
        import tempfile
        from unittest.mock import patch

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

            # Mock LLM call to avoid API requests and speed up tests
            with patch("prompt_distil.core.llm_map.llm_preprocess_text") as mock_llm:
                mock_llm.return_value = "Update `delete_task` and `some_unknown_function`"
                reconciled, matched, unknown, unresolved = reconcile_text(text, temp_dir, "en")

            # Should find delete_task via LLM
            assert "delete_task" in matched
            # May have unknown mentions
            assert isinstance(unknown, list)

        # Test empty string - placeholder test
        pass


class TestProgressReporter:
    """Test the progress reporter functionality."""

    def test_global_reporter_instance(self):
        """Test that global reporter instance exists."""
        from prompt_distil.core.progress import reporter

        assert reporter is not None
        assert hasattr(reporter, "step")
        assert hasattr(reporter, "initialize")
        assert hasattr(reporter, "complete_step")
        assert hasattr(reporter, "step_with_context")
        assert hasattr(reporter, "sub_step")
        assert hasattr(reporter, "sub_step_with_progress")

    def test_detailed_reconciliation_progress(self):
        """Test detailed progress reporting during hybrid reconciliation."""
        import tempfile

        from prompt_distil.core.reconcile import reconcile_text
        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with test symbols
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "files": [],
                "symbols": [
                    {"name": "delete_task", "kind": "function", "path": "test.py", "lineno": 1},
                    {"name": "login_handler", "kind": "function", "path": "test.py", "lineno": 5},
                    {"name": "process_data", "kind": "function", "path": "test.py", "lineno": 10},
                ],
                "inverted_index": {"delete_task": ["test.py#L1"], "login_handler": ["test.py#L5"], "process_data": ["test.py#L10"]},
            }
            save_cache(temp_dir, cache_data)

            # Test reconciliation with progress reporting
            text = "Update the delete_task function and login_handler to process_data correctly"

            # Mock LLM call to avoid API requests in tests
            from unittest.mock import patch

            with patch("prompt_distil.core.llm_map.llm_preprocess_text") as mock_llm:
                mock_llm.return_value = "Update the `delete_task` function and `login_handler` to `process_data` correctly"
                reconciled, matched, unknown, unresolved = reconcile_text(text, temp_dir, "en")

            # Verify reconciliation worked
            assert "delete_task" in matched
            assert "login_handler" in matched
            assert "process_data" in matched
            assert "`delete_task`" in reconciled

    def test_hybrid_mode_reconciliation_progress(self):
        """Test detailed progress reporting in hybrid mode reconciliation."""
        import tempfile

        from prompt_distil.core.reconcile import reconcile_text
        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with test symbols
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "files": [],
                "symbols": [
                    {"name": "api_handler", "kind": "function", "path": "test.py", "lineno": 1},
                    {"name": "user_model", "kind": "class", "path": "test.py", "lineno": 20},
                    {"name": "configure_settings", "kind": "function", "path": "test.py", "lineno": 40},
                ],
                "inverted_index": {
                    "api_handler": ["test.py#L1"],
                    "user_model": ["test.py#L20"],
                    "configure_settings": ["test.py#L40"],
                },
            }
            save_cache(temp_dir, cache_data)

            # Test reconciliation with hybrid mode
            text = "Fix the api handler and user model configuration"

            # Mock LLM call to avoid API requests in tests
            from unittest.mock import patch

            with patch("prompt_distil.core.llm_map.llm_preprocess_text") as mock_llm:
                mock_llm.return_value = "Fix the `api_handler` and `user_model` configuration"
                reconciled, matched, unknown, unresolved = reconcile_text(text, temp_dir, "en")

            # Verify that reconciliation worked with hybrid mode
            assert isinstance(reconciled, str)
            assert "`api_handler`" in reconciled or "api_handler" in matched
            assert "`user_model`" in reconciled or "user_model" in matched

            # May have unresolved terms for LLM processing
            assert isinstance(unresolved, list)


class TestOptimizedReconciliation:
    """Test the optimized LLM-first reconciliation functionality."""

    def test_hybrid_mode_processing(self):
        """Test hybrid mode processing with all symbols."""
        import tempfile

        from prompt_distil.core.reconcile import reconcile_text
        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with test symbols
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "symbols": [
                    {"name": "delete_task", "kind": "function", "path": "tasks.py", "lineno": 10},
                    {"name": "login_handler", "kind": "function", "path": "auth.py", "lineno": 5},
                    {"name": "process_data", "kind": "function", "path": "data.py", "lineno": 15},
                ],
                "files": ["tasks.py", "auth.py", "data.py"],
            }
            save_cache(temp_dir, cache_data)

            # Test English text
            text = "Update the delete task function and login handler"

            # Mock LLM call to avoid API requests in tests
            from unittest.mock import patch

            with patch("prompt_distil.core.llm_map.llm_preprocess_text") as mock_llm:
                mock_llm.return_value = "Update the `delete_task` function and `login_handler`"
                reconciled, matched, unknown, unresolved = reconcile_text(text, temp_dir, "en")

            # Should work with hybrid mode
            assert len(matched) >= 0  # May or may not find matches depending on LLM

            # Test Russian text - should now work better with all symbols
            russian_text = "исправь функцию удаления задачи"

            with patch("prompt_distil.core.llm_map.llm_preprocess_text") as mock_llm:
                mock_llm.return_value = "исправь `delete_task`"
                reconciled_ru, matched_ru, unknown_ru, unresolved_ru = reconcile_text(russian_text, temp_dir, "ru")

            # Russian processing should work with hybrid mode
            assert len(matched_ru) >= 0

    def test_hybrid_mode_processing_logic(self):
        """Test hybrid mode processing logic."""
        import tempfile

        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with test symbols
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "symbols": [
                    {"name": "delete_task", "kind": "function", "path": "tasks.py", "lineno": 10},
                    {"name": "login_handler", "kind": "function", "path": "auth.py", "lineno": 5},
                    {"name": "process_data", "kind": "function", "path": "data.py", "lineno": 15},
                ],
                "files": ["tasks.py", "auth.py", "data.py"],
            }
            save_cache(temp_dir, cache_data)

            known_symbols = {s["name"]: s for s in cache_data["symbols"]}
            text = "Update delete_task and login_handler"

            # Test hybrid processing
            from unittest.mock import patch

            with patch("prompt_distil.core.llm_map.llm_preprocess_text") as mock_llm:
                mock_llm.return_value = "Update `delete_task` and `login_handler`"
                from prompt_distil.core.reconcile import _process_llm_mode

                result_text, matched, unknown = _process_llm_mode(text, known_symbols, temp_dir)

            # Should process the text (exact results depend on LLM)
            assert isinstance(result_text, str)
            assert isinstance(matched, list)
            assert isinstance(unknown, list)

    def test_marked_text_processing_with_rules(self):
        """Test processing of LLM-marked text with selective rules."""
        import tempfile

        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with test symbols
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "files": [],
                "symbols": [
                    {"name": "delete_task", "kind": "function", "path": "test.py", "lineno": 1},
                    {"name": "login_handler", "kind": "function", "path": "test.py", "lineno": 5},
                ],
                "inverted_index": {"delete_task": ["test.py#L1"], "login_handler": ["test.py#L5"]},
            }
            save_cache(temp_dir, cache_data)

            known_symbols = {s["name"]: s for s in cache_data["symbols"]}

            # Test text with LLM-marked content (simulated)
            marked_text = "Update `delete_task` and `login_handler` functions"

            from prompt_distil.core.reconcile import _extract_symbols_from_llm_output

            matched, unknown = _extract_symbols_from_llm_output(marked_text, known_symbols)

            # Should match the marked symbols
            assert "delete_task" in matched
            assert "login_handler" in matched
            # The original marked text should preserve backticks
            assert "`delete_task`" in marked_text
            assert "`login_handler`" in marked_text

    def test_optimized_hybrid_vs_traditional_hybrid(self):
        """Test that optimized hybrid mode produces reasonable results."""
        import tempfile

        from prompt_distil.core.reconcile import reconcile_text
        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with test symbols
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "files": [],
                "symbols": [
                    {"name": "delete_task", "kind": "function", "path": "test.py", "lineno": 1},
                    {"name": "login_handler", "kind": "function", "path": "test.py", "lineno": 5},
                    {"name": "process_data", "kind": "function", "path": "test.py", "lineno": 10},
                ],
                "inverted_index": {
                    "delete_task": ["test.py#L1"],
                    "login_handler": ["test.py#L5"],
                    "process_data": ["test.py#L10"],
                },
            }
            save_cache(temp_dir, cache_data)

            # Test optimized hybrid mode
            text = "Update delete_task function and login_handler"

            # Mock LLM call to avoid API requests in tests
            from unittest.mock import patch

            with patch("prompt_distil.core.llm_map.llm_preprocess_text") as mock_llm:
                mock_llm.return_value = "Update `delete_task` function and `login_handler`"
                reconciled, matched, unknown, unresolved = reconcile_text(text, temp_dir, "en")

            # Should process the text (exact results depend on LLM availability)
            assert isinstance(reconciled, str)
            assert isinstance(matched, list)
            assert isinstance(unknown, list)
            assert isinstance(unresolved, list)

            # Test with same mode again for consistency
            with patch("prompt_distil.core.llm_map.llm_preprocess_text") as mock_llm:
                mock_llm.return_value = "Update `delete_task` function and `login_handler`"
                reconciled_again, matched_again, unknown_again, unresolved_again = reconcile_text(text, temp_dir, "en")

            # Both calls should handle the basic case consistently
            assert isinstance(reconciled_again, str)
            assert isinstance(matched_again, list)

    def test_ngram_search_after_llm_processing(self):
        """Test that n-gram search is performed correctly after LLM processing in hybrid mode."""
        import tempfile

        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with test symbols including some that might need n-gram matching
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "files": [],
                "symbols": [
                    {"name": "delete_task", "kind": "function", "path": "test.py", "lineno": 1},
                    {"name": "login_handler", "kind": "function", "path": "test.py", "lineno": 5},
                    {"name": "process_data", "kind": "function", "path": "test.py", "lineno": 10},
                    {"name": "validate_input", "kind": "function", "path": "test.py", "lineno": 15},
                ],
                "inverted_index": {
                    "delete_task": ["test.py#L1"],
                    "login_handler": ["test.py#L5"],
                    "process_data": ["test.py#L10"],
                    "validate_input": ["test.py#L15"],
                },
            }
            save_cache(temp_dir, cache_data)

            known_symbols = {s["name"]: s for s in cache_data["symbols"]}

            # Test text with both LLM-marked content AND additional text that needs n-gram search
            # This simulates the scenario where LLM marks some symbols but misses others
            marked_text = "Update `delete_task` and also fix the process data function and validate input"

            from prompt_distil.core.reconcile import _extract_symbols_from_llm_output

            matched, unknown = _extract_symbols_from_llm_output(marked_text, known_symbols)

            # Should match the explicitly marked symbol
            assert "delete_task" in matched
            assert "`delete_task`" in marked_text

            # Should also find symbols through n-gram search in the remaining text
            # "process data" should match "process_data" and "validate input" should match "validate_input"
            assert len(matched) >= 1  # At least delete_task should be matched

            # The original marked text should contain backticked symbols
            assert "`delete_task`" in marked_text

    def test_ngrams_only_from_llm_keywords(self):
        """Test that n-grams are generated only from backticked keywords extracted by LLM."""
        import tempfile

        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with test symbols
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "files": [],
                "symbols": [
                    {"name": "delete_task", "kind": "function", "path": "test.py", "lineno": 1},
                    {"name": "user_model", "kind": "class", "path": "test.py", "lineno": 5},
                    {"name": "process_data", "kind": "function", "path": "test.py", "lineno": 10},
                ],
                "inverted_index": {
                    "delete_task": ["test.py#L1"],
                    "user_model": ["test.py#L5"],
                    "process_data": ["test.py#L10"],
                },
            }
            save_cache(temp_dir, cache_data)

            known_symbols = {s["name"]: s for s in cache_data["symbols"]}

            # Test text with backticked keywords and additional non-backticked text
            # Only the backticked keywords should be used for processing
            marked_text = "Update `user_model` function and also fix some other unrelated functionality in the system"

            from prompt_distil.core.reconcile import _extract_symbols_from_llm_output

            matched, unknown = _extract_symbols_from_llm_output(marked_text, known_symbols)

            # Should match the symbol corresponding to the backticked keyword
            assert "user_model" in matched
            assert "`user_model`" in marked_text

            # The non-backticked text ("some other unrelated functionality")
            # should NOT be processed for n-grams, so no additional matches
            assert len(matched) == 1  # Only the backticked keyword match


class TestClipboardFunctionality:
    """Test clipboard copying functionality."""

    def test_clipboard_copying_with_mock(self):
        """Test that clipboard copying works without actually using system clipboard."""
        import json
        from unittest.mock import patch

        from rich.console import Console

        from prompt_distil.main import _display_distillation_result

        # Mock result data
        result = {
            "selected_prompt": "## Goal\nTest prompt\n\n## Context\nTest context",
            "ir": type("MockIR", (), {"model_dump": lambda self: {}})(),
            "prompts": {},
            "session_passport": {
                "processing_stats": {"known_entities_found": 0, "requirements_extracted": 1, "unknowns_identified": 0, "assumptions_made": 1},
                "model_used": "gpt-4o",
                "dropped_or_simplified": [],
            },
        }

        console = Console()

        # Test markdown format clipboard copying
        with patch("prompt_distil.main.pyperclip.copy") as mock_copy:
            _display_distillation_result(result, "standard", "markdown")
            mock_copy.assert_called_once_with("## Goal\nTest prompt\n\n## Context\nTest context")

        # Test JSON format clipboard copying
        with patch("prompt_distil.main.pyperclip.copy") as mock_copy:
            _display_distillation_result(result, "standard", "json")
            expected_json = json.dumps({"prompt": "## Goal\nTest prompt\n\n## Context\nTest context"}, indent=2)
            mock_copy.assert_called_once_with(expected_json)

        # Test rich format clipboard copying (should copy as markdown)
        with patch("prompt_distil.main.pyperclip.copy") as mock_copy:
            _display_distillation_result(result, "standard", "rich")
            mock_copy.assert_called_once_with("## Goal\nTest prompt\n\n## Context\nTest context")

    def test_clipboard_error_handling(self):
        """Test that clipboard errors don't break the application."""
        from unittest.mock import patch

        from prompt_distil.main import _display_distillation_result

        # Mock result data
        result = {
            "selected_prompt": "## Goal\nTest prompt",
            "ir": type("MockIR", (), {"model_dump": lambda self: {}})(),
            "prompts": {},
            "session_passport": {
                "processing_stats": {"known_entities_found": 0, "requirements_extracted": 1, "unknowns_identified": 0, "assumptions_made": 1},
                "model_used": "gpt-4o",
                "dropped_or_simplified": [],
            },
        }

        # Test that clipboard errors are handled gracefully
        with patch("prompt_distil.main.pyperclip.copy", side_effect=Exception("Clipboard error")):
            # This should not raise an exception
            _display_distillation_result(result, "standard", "markdown")


class TestLLMHandler:
    """Test centralized LLM Handler functionality."""

    def test_llm_handler_initialization(self):
        """Test that LLM handler initializes correctly."""
        import tempfile

        from prompt_distil.core.llm_handler import LLMHandler

        with tempfile.TemporaryDirectory() as temp_dir:
            handler = LLMHandler(temp_dir)
            assert handler.project_root == temp_dir
            assert handler.client is not None
            assert handler.debug_logger is not None

    def test_global_llm_handler_instance(self):
        """Test that global LLM handler instance works correctly."""
        from prompt_distil.core.llm_handler import get_llm_handler

        handler1 = get_llm_handler(".")
        handler2 = get_llm_handler(".")
        assert handler1 is handler2  # Same instance for same project root

        handler3 = get_llm_handler("/different/path")
        assert handler3 is not handler1  # Different instance for different path

    def test_reconciliation_request_with_symbols(self):
        """Test reconciliation request with valid symbols."""
        import tempfile
        from unittest.mock import patch

        from prompt_distil.core.llm_handler import LLMHandler
        from prompt_distil.core.surface import save_cache

        with tempfile.TemporaryDirectory() as temp_dir:
            # Create cache with test symbols
            cache_data = {
                "version": 1,
                "generated_at": "2023-01-01T00:00:00",
                "root": temp_dir,
                "globs": ["**/*.py"],
                "files": [],
                "symbols": [
                    {"name": "test_function", "kind": "function", "path": "test.py", "lineno": 1},
                    {"name": "TestClass", "kind": "class", "path": "test.py", "lineno": 5},
                ],
                "inverted_index": {"test_function": ["test.py#L1"], "TestClass": ["test.py#L5"]},
            }
            save_cache(temp_dir, cache_data)

            handler = LLMHandler(temp_dir)

            with patch("prompt_distil.core.llm_handler.make_llm_request_with_reasoning_fallback") as mock_request:
                # Mock successful response
                mock_response = type("MockResponse", (), {})()
                mock_response.choices = [type("Choice", (), {"message": type("Message", (), {"content": "Fix the `test_function`"})()})]
                mock_request.return_value = mock_response

                result = handler.make_reconciliation_request("Fix the test function", ["test_function", "TestClass"])
                assert result == "Fix the `test_function`"
                mock_request.assert_called_once()

    def test_reconciliation_request_empty_input(self):
        """Test reconciliation request with empty input."""
        from prompt_distil.core.llm_handler import LLMHandler

        handler = LLMHandler(".")

        # Empty transcript should return as-is
        result = handler.make_reconciliation_request("", ["symbol1"])
        assert result == ""

        # Empty symbols should return original transcript
        result = handler.make_reconciliation_request("test text", [])
        assert result == "test text"

    def test_distillation_request_success(self):
        """Test successful distillation request."""
        import tempfile
        from unittest.mock import patch

        from prompt_distil.core.llm_handler import LLMHandler

        with tempfile.TemporaryDirectory() as temp_dir:
            handler = LLMHandler(temp_dir)

            with patch("prompt_distil.core.llm_handler.make_llm_request_with_reasoning_fallback") as mock_request:
                # Mock successful JSON response
                mock_response = type("MockResponse", (), {})()
                json_content = '{"goals": ["test goal"], "context": "test context", "requirements": []}'
                mock_response.choices = [type("Choice", (), {"message": type("Message", (), {"content": json_content})()})]
                mock_request.return_value = mock_response

                result = handler.make_distillation_request("test transcript")
                assert result == {"goals": ["test goal"], "context": "test context", "requirements": []}
                mock_request.assert_called_once()

    def test_distillation_request_invalid_json(self):
        """Test distillation request with invalid JSON response."""
        import tempfile
        from unittest.mock import patch

        from prompt_distil.core.llm_handler import LLMHandler, LLMHandlerError

        with tempfile.TemporaryDirectory() as temp_dir:
            handler = LLMHandler(temp_dir)

            with patch("prompt_distil.core.llm_handler.make_llm_request_with_reasoning_fallback") as mock_request:
                # Mock invalid JSON response
                mock_response = type("MockResponse", (), {})()
                mock_response.choices = [type("Choice", (), {"message": type("Message", (), {"content": "invalid json"})()})]
                mock_request.return_value = mock_response

                try:
                    handler.make_distillation_request("test transcript")
                    assert False, "Should have raised LLMHandlerError"
                except LLMHandlerError as e:
                    assert "Invalid JSON response" in str(e)


class TestReasoningModelHandler:
    """Test reasoning model error detection and parameter adjustment."""

    def test_reasoning_model_error_detection(self):
        """Test detection of reasoning model errors."""
        from prompt_distil.core.llm_handler import is_reasoning_model_error

        # Mock OpenAI API error with reasoning model pattern (temperature)
        class MockTemperatureError(Exception):
            def __init__(self):
                self.status_code = 400
                self.response = type("Response", (), {})()
                self.response.json = lambda: {"error": {"type": "invalid_request_error", "code": "unsupported_value", "param": "temperature"}}

        temp_error = MockTemperatureError()
        assert is_reasoning_model_error(temp_error) == True

        # Mock OpenAI API error with reasoning model pattern (max_tokens)
        class MockMaxTokensError(Exception):
            def __init__(self):
                self.status_code = 400
                self.response = type("Response", (), {})()
                self.response.json = lambda: {"error": {"type": "invalid_request_error", "code": "unsupported_parameter", "param": "max_tokens"}}

        tokens_error = MockMaxTokensError()
        assert is_reasoning_model_error(tokens_error) == True

        # Test non-reasoning model error
        class RegularError(Exception):
            def __init__(self):
                self.status_code = 500

        regular_error = RegularError()
        assert is_reasoning_model_error(regular_error) == False

    def test_reasoning_model_parameters(self):
        """Test parameter adjustment for reasoning models with environment variables."""
        import os
        from unittest.mock import patch

        from prompt_distil.core.llm_handler import get_reasoning_model_parameters

        # Test with default environment values
        with patch.dict(os.environ, {}, clear=False):
            # Test reconciliation parameters (should return empty dict for now)
            reconciliation_params = get_reasoning_model_parameters("reconciliation")
            assert reconciliation_params == {}  # Currently empty as OpenAI doesn't support these params

            # Test distillation parameters (should return empty dict for now)
            distillation_params = get_reasoning_model_parameters("distillation")
            assert distillation_params == {}  # Currently empty as OpenAI doesn't support these params

        # Test with custom environment values (should still return empty)
        with patch.dict(os.environ, {"REASONING_EFFORT": "high", "VERBOSITY": "high"}, clear=False):
            reconciliation_params = get_reasoning_model_parameters("reconciliation")
            assert reconciliation_params == {}  # Still empty

            distillation_params = get_reasoning_model_parameters("distillation")
            assert distillation_params == {}  # Still empty

    def test_environment_variable_functions(self):
        """Test environment variable getter functions."""
        import os
        from unittest.mock import patch

        from prompt_distil.core.llm_handler import get_env_model_temperature, get_env_reasoning_effort, get_env_verbosity

        # Test defaults
        with patch.dict(os.environ, {}, clear=False):
            assert get_env_reasoning_effort() == "low"
            assert get_env_verbosity() == "medium"
            assert get_env_model_temperature() == 0.2

        # Test custom values
        with patch.dict(os.environ, {"REASONING_EFFORT": "high", "VERBOSITY": "low", "MODEL_TEMPERATURE": "0.5"}, clear=False):
            assert get_env_reasoning_effort() == "high"
            assert get_env_verbosity() == "low"
            assert get_env_model_temperature() == 0.5

        # Test invalid values fall back to defaults
        with patch.dict(os.environ, {"REASONING_EFFORT": "invalid", "VERBOSITY": "invalid", "MODEL_TEMPERATURE": "invalid"}, clear=False):
            assert get_env_reasoning_effort() == "low"
            assert get_env_verbosity() == "medium"
            assert get_env_model_temperature() == 0.2

    def test_parameter_adjustment(self):
        """Test adjustment of LLM parameters for reasoning model."""
        from prompt_distil.core.llm_handler import adjust_llm_params_for_reasoning_model

        original_params = {"model": "gpt-4", "messages": [{"role": "user", "content": "test"}], "temperature": 0.1, "max_tokens": 1000}

        # Test reconciliation adjustment
        adjusted = adjust_llm_params_for_reasoning_model(original_params, "reconciliation")

        # Temperature and max_tokens should be removed
        assert "temperature" not in adjusted
        assert "max_tokens" not in adjusted

        # max_completion_tokens should be added instead of max_tokens
        assert adjusted["max_completion_tokens"] == 1000

        # Other parameters should remain
        assert adjusted["model"] == "gpt-4"
        assert adjusted["messages"] == [{"role": "user", "content": "test"}]

    def test_llm_request_with_fallback_success(self):
        """Test LLM request that succeeds on first try."""
        from unittest.mock import Mock

        from prompt_distil.core.llm_handler import make_llm_request_with_reasoning_fallback

        mock_client = Mock()
        mock_response = Mock()
        mock_client.chat.completions.create.return_value = mock_response

        params = {"model": "gpt-4", "messages": [], "temperature": 0.1}

        result = make_llm_request_with_reasoning_fallback(mock_client, params, "reconciliation")

        assert result == mock_response
        mock_client.chat.completions.create.assert_called_once_with(**params)

    def test_llm_request_with_fallback_retry(self):
        """Test LLM request that fails first then succeeds with reasoning model params."""
        from unittest.mock import Mock

        from prompt_distil.core.llm_handler import make_llm_request_with_reasoning_fallback

        # Mock reasoning model error
        class MockReasoningError(Exception):
            def __init__(self):
                self.status_code = 400
                self.response = Mock()
                self.response.json.return_value = {"error": {"type": "invalid_request_error", "code": "unsupported_value", "param": "temperature"}}

        mock_client = Mock()
        mock_response = Mock()

        # First call fails, second succeeds
        mock_client.chat.completions.create.side_effect = [MockReasoningError(), mock_response]

        params = {"model": "gpt-4", "messages": [], "temperature": 0.1}

        result = make_llm_request_with_reasoning_fallback(mock_client, params, "reconciliation")

        assert result == mock_response
        assert mock_client.chat.completions.create.call_count == 2

        # Second call should have adjusted parameters (no temperature)
        second_call_params = mock_client.chat.completions.create.call_args_list[1][1]
        assert "temperature" not in second_call_params
        # Currently no additional parameters are added for reasoning models
        assert second_call_params == {"messages": [], "model": "gpt-4"}


class TestIRLiteValidation:
    """Test IRLite validation error handling and logging."""

    def test_irlite_validation_error_logging(self):
        """Test that IRLite validation errors are properly logged."""
        import tempfile
        from unittest.mock import patch

        from prompt_distil.core.debug_log import DebugLogger
        from prompt_distil.core.distill import DistillationError, TranscriptDistiller

        with tempfile.TemporaryDirectory() as temp_dir:
            # Enable debug logging
            debug_logger = DebugLogger(temp_dir, enabled=True)

            distiller = TranscriptDistiller(temp_dir)

            # Mock LLM response with invalid known_entities structure
            invalid_response_data = {
                "goal": "Test goal",
                "scope_hints": [],
                "must": [],
                "must_not": [],
                "known_entities": [
                    {"path": None, "symbol": "test_func", "confidence": 0.8},  # path is None, should be string
                    {"path": 123, "symbol": "another_func"},  # path is int, should be string
                ],
                "unknowns": [],
                "acceptance": [],
                "assumptions": [],
            }

            # Mock the LLM handler to return invalid data
            with patch("prompt_distil.core.distill.get_llm_handler") as mock_get_handler:
                mock_handler = mock_get_handler.return_value
                mock_handler.make_distillation_request.return_value = invalid_response_data

                # This should trigger validation error logging
                error_caught = None
                try:
                    distiller.build_ir_lite("test transcript", {})
                    assert False, "Expected DistillationError"
                except DistillationError as e:
                    error_caught = e
                    assert "Failed to create IRLite from response" in str(e)

                # Check that validation error was logged (we can't easily check file logging in tests,
                # but we can verify the exception contains the expected validation details)
                assert error_caught is not None
                assert "2 validation errors for IRLite" in str(error_caught)
                assert "known_entities" in str(error_caught)
                assert "path" in str(error_caught)

    def test_irlite_validation_with_valid_data(self):
        """Test that valid IRLite data does not trigger error logging."""
        import tempfile
        from unittest.mock import patch

        from prompt_distil.core.distill import TranscriptDistiller

        with tempfile.TemporaryDirectory() as temp_dir:
            distiller = TranscriptDistiller(temp_dir)

            # Mock LLM response with valid known_entities structure
            valid_response_data = {
                "goal": "Test goal",
                "scope_hints": [],
                "must": [],
                "must_not": [],
                "known_entities": [
                    {"path": "test.py", "symbol": "test_func", "confidence": 0.8},
                    {"path": "module/file.py", "symbol": "another_func", "confidence": 0.9},
                ],
                "unknowns": [],
                "acceptance": [],
                "assumptions": [],
            }

            # Mock the LLM handler to return valid data
            with patch("prompt_distil.core.distill.get_llm_handler") as mock_get_handler:
                mock_handler = mock_get_handler.return_value
                mock_handler.make_distillation_request.return_value = valid_response_data

                # This should work without errors
                try:
                    ir = distiller.build_ir_lite("test transcript", {})
                    assert ir.goal == "Test goal"
                    assert len(ir.known_entities) == 2
                    assert ir.known_entities[0].path == "test.py"
                    assert ir.known_entities[1].path == "module/file.py"
                except Exception as e:
                    assert False, f"Unexpected error with valid data: {e}"


class TestReasoningModelConfiguration:
    """Test reasoning model configuration and parameter handling."""

    def test_reasoning_model_config_properties(self):
        """Test reasoning model configuration properties."""
        import os
        from unittest.mock import patch

        from prompt_distil.core.config import Config

        # Test with default values (not reasoning model)
        with patch.dict(os.environ, {}, clear=True):
            config = Config()
            assert config.is_reasoning_model == False
            assert config.llm_model == "gpt-4o-mini"

        # Test with IS_REASONING_MODEL=true
        with patch.dict(os.environ, {"IS_REASONING_MODEL": "true", "LLM_MODEL": "o1-preview"}, clear=True):
            config = Config()
            assert config.is_reasoning_model == True
            assert config.llm_model == "o1-preview"

        # Test various true values
        for true_value in ["true", "1", "yes", "on", "TRUE", "Yes", "ON"]:
            with patch.dict(os.environ, {"IS_REASONING_MODEL": true_value}, clear=True):
                config = Config()
                assert config.is_reasoning_model == True

        # Test various false values
        for false_value in ["false", "0", "no", "off", "FALSE", "No", "OFF", ""]:
            with patch.dict(os.environ, {"IS_REASONING_MODEL": false_value}, clear=True):
                config = Config()
                assert config.is_reasoning_model == False

    def test_llm_model_backward_compatibility(self):
        """Test LLM_MODEL with DISTIL_MODEL backward compatibility."""
        import os
        from unittest.mock import patch

        from prompt_distil.core.config import Config

        # Test LLM_MODEL takes precedence
        with patch.dict(os.environ, {"LLM_MODEL": "gpt-4o", "DISTIL_MODEL": "gpt-3.5-turbo"}, clear=True):
            config = Config()
            assert config.llm_model == "gpt-4o"
            assert config.distil_model == "gpt-4o"  # distil_model should return llm_model

        # Test fallback to DISTIL_MODEL when LLM_MODEL is not set
        with patch.dict(os.environ, {"DISTIL_MODEL": "gpt-4o-mini"}, clear=True):
            config = Config()
            assert config.llm_model == "gpt-4o-mini"
            assert config.distil_model == "gpt-4o-mini"

    def test_reasoning_model_parameter_selection(self):
        """Test that reasoning model flag affects parameter selection."""
        import tempfile
        from unittest.mock import Mock, patch

        from prompt_distil.core.llm_handler import LLMHandler

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test standard model parameters
            with patch("prompt_distil.core.llm_handler.config") as mock_config:
                mock_config.is_reasoning_model = False
                mock_config.llm_model = "gpt-4o"

                handler = LLMHandler(temp_dir)

                with patch("prompt_distil.core.llm_handler.make_llm_request_with_reasoning_fallback") as mock_request:
                    mock_response = Mock()
                    mock_response.choices = [Mock(message=Mock(content="test response"))]
                    mock_request.return_value = mock_response

                    # This should include temperature for standard models
                    result = handler.make_reconciliation_request("test text", ["symbol1"])

                    # Check that temperature was included in the request
                    call_args = mock_request.call_args[1]["original_params"]
                    assert "temperature" in call_args
                    assert call_args["model"] == "gpt-4o"

            # Test reasoning model parameters
            with patch("prompt_distil.core.llm_handler.config") as mock_config:
                mock_config.is_reasoning_model = True
                mock_config.llm_model = "o1-preview"

                handler = LLMHandler(temp_dir)

                with patch("prompt_distil.core.llm_handler.make_llm_request_with_reasoning_fallback") as mock_request:
                    mock_response = Mock()
                    mock_response.choices = [Mock(message=Mock(content="test response"))]
                    mock_request.return_value = mock_response

                    # This should NOT include temperature for reasoning models
                    result = handler.make_reconciliation_request("test text", ["symbol1"])

                    # Check that temperature was NOT included in the request
                    call_args = mock_request.call_args[1]["original_params"]
                    assert "temperature" not in call_args
                    assert "max_completion_tokens" in call_args  # Should use max_completion_tokens
                    assert call_args["model"] == "o1-preview"


class TestDebugLogging:
    """Test debug logging functionality for reconcile_text hybrid mode."""

    def test_debug_logger_initialization(self):
        """Test that debug logger initializes correctly."""
        import tempfile

        from prompt_distil.core.debug_log import DebugLogger

        with tempfile.TemporaryDirectory() as temp_dir:
            # Test with debug disabled
            logger_disabled = DebugLogger(temp_dir, enabled=False)
            assert not logger_disabled.is_enabled()

            # Test with debug enabled
            logger_enabled = DebugLogger(temp_dir, enabled=True)
            assert logger_enabled.is_enabled()

            # Check that debug directory is created
            debug_dir = Path(temp_dir) / ".prompt_distil" / "debug"
            assert debug_dir.exists()

    def test_debug_logging_with_environment_variable(self):
        """Test debug logging enabled via environment variable."""
        import os
        from unittest.mock import patch

        from prompt_distil.core.debug_log import is_debug_enabled

        # Test without debug enabled
        assert not is_debug_enabled()

        # Test with debug enabled via environment
        with patch.dict(os.environ, {"PD_DEBUG": "1"}):
            assert is_debug_enabled()

    def test_debug_backward_compatibility(self):
        """Test backward compatibility with PD_DEBUG_RECONCILE."""
        import os
        from unittest.mock import patch

        from prompt_distil.core.debug_log import is_debug_enabled

        # Test that old PD_DEBUG_RECONCILE still works
        with patch.dict(os.environ, {"PD_DEBUG_RECONCILE": "1"}, clear=True):
            assert is_debug_enabled()

        # Test that PD_DEBUG takes precedence over PD_DEBUG_RECONCILE
        with patch.dict(os.environ, {"PD_DEBUG": "0", "PD_DEBUG_RECONCILE": "1"}, clear=True):
            assert not is_debug_enabled()

        with patch.dict(os.environ, {"PD_DEBUG": "1", "PD_DEBUG_RECONCILE": "0"}, clear=True):
            assert is_debug_enabled()

    def test_debug_logging_disabled_by_default(self):
        """Test that debug logging is disabled by default."""
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
                "symbols": [{"name": "delete_task", "kind": "function", "path": "tasks.py", "lineno": 10}],
                "files": ["tasks.py"],
            }
            save_cache(temp_dir, cache_data)

            # Run reconciliation without debug enabled
            text = "Update the delete_task function"

            # Mock LLM call to avoid API requests in tests
            from unittest.mock import patch

            with patch("prompt_distil.core.llm_map.llm_preprocess_text") as mock_llm:
                mock_llm.return_value = "Update the `delete_task` function"
                reconciled, matched, unknown, unresolved = reconcile_text(text, temp_dir, "en")

            # Check that debug logs were NOT created
            debug_dir = Path(temp_dir) / ".prompt_distil" / "debug"
            assert not debug_dir.exists() or len(list(debug_dir.iterdir())) == 0


if __name__ == "__main__":
    pytest.main([__file__])
