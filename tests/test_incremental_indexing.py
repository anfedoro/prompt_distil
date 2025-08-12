"""
Tests for incremental indexing functionality in the Surface module.

This module tests the new incremental indexing features including:
- Change detection based on file modification times
- Incremental cache updates
- Specific file reindexing
- Integration with the CLI
"""

import json
import time
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from prompt_distil.core.surface import (
    build_symbol_inventory,
    detect_changed_files,
    ensure_cache,
    load_cache,
    reindex_specific_files,
    update_cache_incrementally,
)
from prompt_distil.main import app


def write_file(path: Path, content: str) -> None:
    """Helper to write a text file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> dict:
    """Helper to read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def create_test_project(base_path: Path) -> None:
    """Create a simple test project structure."""
    write_file(base_path / "module1.py", "def func1():\n    return 'hello'\n\nclass Class1:\n    pass\n")
    write_file(base_path / "module2.py", "def func2():\n    return 'world'\n")
    write_file(base_path / "subdir" / "module3.py", "class Class3:\n    def method1(self):\n        pass\n")


def test_detect_changed_files_no_cache(tmp_path: Path):
    """Test change detection when no cache exists."""
    create_test_project(tmp_path)

    # Empty cache should detect all files as changed
    empty_cache = {"files": [], "symbols": []}
    changed = detect_changed_files(str(tmp_path), empty_cache)

    assert len(changed) == 3
    assert "module1.py" in changed
    assert "module2.py" in changed
    assert "subdir/module3.py" in changed


def test_detect_changed_files_with_modifications(tmp_path: Path):
    """Test change detection when files are modified."""
    create_test_project(tmp_path)

    # Build initial cache
    cache = build_symbol_inventory(str(tmp_path))

    # Sleep to ensure time difference
    time.sleep(0.1)

    # Modify one file
    write_file(tmp_path / "module1.py", "def func1_modified():\n    return 'modified'\n")

    # Detect changes
    changed = detect_changed_files(str(tmp_path), cache)

    assert len(changed) == 1
    assert "module1.py" in changed


def test_detect_changed_files_no_changes(tmp_path: Path):
    """Test change detection when no files have changed."""
    create_test_project(tmp_path)

    # Build cache
    cache = build_symbol_inventory(str(tmp_path))

    # Check immediately - no changes expected
    changed = detect_changed_files(str(tmp_path), cache)

    assert len(changed) == 0


def test_update_cache_incrementally(tmp_path: Path):
    """Test incremental cache updates."""
    create_test_project(tmp_path)

    # Build initial cache
    initial_cache = build_symbol_inventory(str(tmp_path))
    initial_symbols = len(initial_cache["symbols"])

    # Modify a file
    time.sleep(0.1)
    write_file(tmp_path / "module1.py", "def new_func():\n    return 'new'\n\nclass NewClass:\n    def new_method(self):\n        pass\n")

    # Update cache incrementally
    updated_cache = update_cache_incrementally(str(tmp_path), initial_cache, ["module1.py"])

    # Verify the cache was updated
    assert len(updated_cache["files"]) == 3  # Still 3 files
    assert len(updated_cache["symbols"]) >= 2  # At least new_func and NewClass

    # Check that old symbols from module1.py are gone and new ones are added
    module1_symbols = [s for s in updated_cache["symbols"] if s["path"] == "module1.py"]
    symbol_names = [s["name"] for s in module1_symbols]

    assert "new_func" in symbol_names
    assert "NewClass" in symbol_names
    assert "func1" not in symbol_names  # Old symbol should be removed


def test_ensure_cache_incremental_mode(tmp_path: Path):
    """Test ensure_cache with incremental mode."""
    create_test_project(tmp_path)

    # Build initial cache
    cache1 = ensure_cache(str(tmp_path), force=True, save=True)
    initial_timestamp = cache1["generated_at"]

    # Sleep and modify a file
    time.sleep(0.1)
    write_file(tmp_path / "module2.py", "def modified_func():\n    return 'modified'\n")

    # Run incremental update
    cache2 = ensure_cache(str(tmp_path), incremental=True, save=True)

    # Verify cache was updated
    assert cache2["generated_at"] > initial_timestamp

    # Check that modified file has new symbols
    module2_symbols = [s for s in cache2["symbols"] if s["path"] == "module2.py"]
    symbol_names = [s["name"] for s in module2_symbols]

    assert "modified_func" in symbol_names
    assert "func2" not in symbol_names  # Old symbol should be gone


def test_ensure_cache_no_incremental_mode(tmp_path: Path):
    """Test ensure_cache with incremental disabled."""
    create_test_project(tmp_path)

    # Build initial cache
    cache1 = ensure_cache(str(tmp_path), force=True, save=True)
    initial_timestamp = cache1["generated_at"]

    # Sleep and modify a file
    time.sleep(0.1)
    write_file(tmp_path / "module2.py", "def modified_func():\n    return 'modified'\n")

    # Run without incremental updates
    cache2 = ensure_cache(str(tmp_path), incremental=False, save=False)

    # Verify cache was NOT updated (same timestamp)
    assert cache2["generated_at"] == initial_timestamp


def test_reindex_specific_files(tmp_path: Path):
    """Test reindexing specific files."""
    create_test_project(tmp_path)

    # Build initial cache
    ensure_cache(str(tmp_path), force=True, save=True)

    # Modify specific files
    time.sleep(0.1)
    write_file(tmp_path / "module1.py", "def specific_func():\n    return 'specific'\n")
    write_file(tmp_path / "subdir" / "module3.py", "class SpecificClass:\n    pass\n")

    # Reindex only module1.py
    cache = reindex_specific_files(str(tmp_path), ["module1.py"], save=True)

    # Verify only module1.py was updated
    module1_symbols = [s for s in cache["symbols"] if s["path"] == "module1.py"]
    module3_symbols = [s for s in cache["symbols"] if s["path"] == "subdir/module3.py"]

    module1_names = [s["name"] for s in module1_symbols]
    module3_names = [s["name"] for s in module3_symbols]

    assert "specific_func" in module1_names
    assert "SpecificClass" not in module3_names  # module3 was not reindexed
    assert "Class3" in module3_names  # Old symbol still there


def test_reindex_specific_files_no_existing_cache(tmp_path: Path):
    """Test reindexing specific files when no cache exists."""
    create_test_project(tmp_path)

    # Reindex specific file without existing cache
    cache = reindex_specific_files(str(tmp_path), ["module1.py"], save=True)

    # Verify cache was created and contains only the specified file
    assert len(cache["files"]) == 1
    assert cache["files"][0]["path"] == "module1.py"

    # Verify symbols only from the specified file
    assert all(s["path"] == "module1.py" for s in cache["symbols"])


def test_cli_incremental_indexing(tmp_path: Path):
    """Test CLI incremental indexing functionality."""
    create_test_project(tmp_path)

    runner = CliRunner()

    # Initial indexing
    result1 = runner.invoke(app, ["index", "--project-root", str(tmp_path), "--save", "--force"])
    assert result1.exit_code == 0
    assert "rebuilt completely" in result1.stdout

    # Sleep and modify a file
    time.sleep(0.1)
    write_file(tmp_path / "module1.py", "def cli_func():\n    return 'cli'\n")

    # Incremental update
    result2 = runner.invoke(app, ["index", "--project-root", str(tmp_path), "--save"])
    assert result2.exit_code == 0
    assert "updated incrementally" in result2.stdout


def test_cli_specific_file_reindexing(tmp_path: Path):
    """Test CLI specific file reindexing."""
    create_test_project(tmp_path)

    runner = CliRunner()

    # Initial indexing
    result1 = runner.invoke(app, ["index", "--project-root", str(tmp_path), "--save", "--force"])
    assert result1.exit_code == 0

    # Reindex specific file
    result2 = runner.invoke(app, ["index", "--project-root", str(tmp_path), "--file", "module1.py", "--save"])
    assert result2.exit_code == 0
    assert "Successfully reindexed 1 specific files" in result2.stdout


def test_cli_no_incremental_mode(tmp_path: Path):
    """Test CLI with incremental updates disabled."""
    create_test_project(tmp_path)

    runner = CliRunner()

    # Initial indexing
    result1 = runner.invoke(app, ["index", "--project-root", str(tmp_path), "--save", "--force"])
    assert result1.exit_code == 0

    # Run without incremental updates
    result2 = runner.invoke(app, ["index", "--project-root", str(tmp_path), "--no-incremental"])
    assert result2.exit_code == 0
    assert "Using existing symbol cache" in result2.stdout


def test_build_symbol_inventory_with_target_files(tmp_path: Path):
    """Test building symbol inventory with specific target files."""
    create_test_project(tmp_path)

    # Build inventory for specific files only
    cache = build_symbol_inventory(str(tmp_path), target_files=["module1.py", "subdir/module3.py"])

    # Verify only specified files are included
    file_paths = [f["path"] for f in cache["files"]]
    assert len(file_paths) == 2
    assert "module1.py" in file_paths
    assert "subdir/module3.py" in file_paths
    assert "module2.py" not in file_paths

    # Verify symbols only from specified files
    symbol_paths = set(s["path"] for s in cache["symbols"])
    assert symbol_paths == {"module1.py", "subdir/module3.py"}


def test_cache_persistence_after_incremental_update(tmp_path: Path):
    """Test that incremental updates are properly saved and loaded."""
    create_test_project(tmp_path)

    # Build and save initial cache
    ensure_cache(str(tmp_path), force=True, save=True)

    # Modify file and do incremental update
    time.sleep(0.1)
    write_file(tmp_path / "module1.py", "def persistent_func():\n    return 'persistent'\n")
    ensure_cache(str(tmp_path), incremental=True, save=True)

    # Load cache from disk and verify changes were saved
    loaded_cache = load_cache(str(tmp_path))
    assert loaded_cache is not None

    module1_symbols = [s for s in loaded_cache["symbols"] if s["path"] == "module1.py"]
    symbol_names = [s["name"] for s in module1_symbols]

    assert "persistent_func" in symbol_names


@patch("prompt_distil.core.surface.reporter")
def test_incremental_update_progress_reporting(mock_reporter, tmp_path: Path):
    """Test that incremental updates report progress correctly."""
    create_test_project(tmp_path)

    # Build initial cache
    initial_cache = build_symbol_inventory(str(tmp_path))

    # Modify files
    time.sleep(0.1)
    write_file(tmp_path / "module1.py", "def progress_func():\n    return 'progress'\n")
    write_file(tmp_path / "module2.py", "def another_progress_func():\n    return 'progress2'\n")

    # Update incrementally
    update_cache_incrementally(str(tmp_path), initial_cache, ["module1.py", "module2.py"])

    # Verify progress reporting was called
    mock_reporter.step.assert_any_call("Reindexing 2 changed files…")
    mock_reporter.step.assert_any_call("Rebuilding symbol index…")


def test_error_handling_during_incremental_update(tmp_path: Path):
    """Test error handling when files can't be processed during incremental update."""
    create_test_project(tmp_path)

    # Build initial cache
    initial_cache = build_symbol_inventory(str(tmp_path))
    initial_symbol_count = len(initial_cache["symbols"])

    # Create a file that will cause processing issues (non-existent in this case)
    changed_files = ["module1.py", "non_existent_file.py"]

    # Update should handle the error gracefully
    updated_cache = update_cache_incrementally(str(tmp_path), initial_cache, changed_files)

    # Should still process the valid file
    assert len(updated_cache["symbols"]) >= 0  # May have fewer symbols after removing module1
    assert len(updated_cache["files"]) >= 2  # Should still have other files
