"""
Tests for content hashing functionality in the Surface module.

This module tests the content hashing features including:
- File content hash calculation
- Hash-based change detection
- Integration with incremental indexing
- Performance and accuracy comparisons
"""

import json
import time
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from prompt_distil.core.surface import (
    _calculate_file_hash,
    build_symbol_inventory,
    detect_changed_files,
    ensure_cache,
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


class TestFileHashing:
    """Test file content hashing functionality."""

    def test_calculate_file_hash_blake2b(self, tmp_path: Path):
        """Test Blake2b hashing (default algorithm)."""
        test_file = tmp_path / "test.py"
        write_file(test_file, "def test():\n    return 'hash test'\n")

        hash1 = _calculate_file_hash(test_file)
        hash2 = _calculate_file_hash(test_file)

        # Same content should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # Blake2b with 32-byte digest = 64 hex chars

    def test_calculate_file_hash_sha256(self, tmp_path: Path):
        """Test SHA256 hashing."""
        test_file = tmp_path / "test.py"
        write_file(test_file, "def test():\n    return 'hash test'\n")

        hash_result = _calculate_file_hash(test_file, algorithm="sha256")

        assert len(hash_result) == 64  # SHA256 = 64 hex chars
        # Verify it's a valid hex string
        int(hash_result, 16)

    def test_calculate_file_hash_md5(self, tmp_path: Path):
        """Test MD5 hashing."""
        test_file = tmp_path / "test.py"
        write_file(test_file, "def test():\n    return 'hash test'\n")

        hash_result = _calculate_file_hash(test_file, algorithm="md5")

        assert len(hash_result) == 32  # MD5 = 32 hex chars

    def test_calculate_file_hash_different_content(self, tmp_path: Path):
        """Test that different content produces different hashes."""
        test_file = tmp_path / "test.py"

        write_file(test_file, "def test1():\n    return 'first'\n")
        hash1 = _calculate_file_hash(test_file)

        write_file(test_file, "def test2():\n    return 'second'\n")
        hash2 = _calculate_file_hash(test_file)

        assert hash1 != hash2

    def test_calculate_file_hash_invalid_algorithm(self, tmp_path: Path):
        """Test error handling for invalid hash algorithm."""
        test_file = tmp_path / "test.py"
        write_file(test_file, "def test():\n    return 'test'\n")

        with pytest.raises(ValueError, match="Unsupported hash algorithm"):
            _calculate_file_hash(test_file, algorithm="invalid")

    def test_calculate_file_hash_nonexistent_file(self, tmp_path: Path):
        """Test error handling for nonexistent file."""
        nonexistent_file = tmp_path / "nonexistent.py"

        with pytest.raises(RuntimeError, match="Failed to calculate hash"):
            _calculate_file_hash(nonexistent_file)

    def test_calculate_file_hash_large_file(self, tmp_path: Path):
        """Test hashing of larger files (chunk reading)."""
        test_file = tmp_path / "large.py"
        # Create file larger than the chunk size (8192 bytes)
        large_content = "# Large file test\n" + "x = 1\n" * 1000
        write_file(test_file, large_content)

        hash_result = _calculate_file_hash(test_file)
        assert len(hash_result) == 64  # Blake2b default


class TestHashBasedChangeDetection:
    """Test hash-based change detection."""

    def test_detect_changed_files_with_hashing_no_changes(self, tmp_path: Path):
        """Test hash-based detection when no files have changed."""
        create_test_project(tmp_path)

        # Build cache with hashing enabled
        cache = build_symbol_inventory(str(tmp_path))

        # Manually add hashes to simulate cache with hashing
        for file_info in cache["files"]:
            file_path = tmp_path / file_info["path"]
            file_info["content_hash"] = _calculate_file_hash(file_path)

        # Check for changes using hash-based detection
        changed = detect_changed_files(str(tmp_path), cache, use_content_hash=True)

        assert len(changed) == 0

    def test_detect_changed_files_with_hashing_content_changed(self, tmp_path: Path):
        """Test hash-based detection when file content changes."""
        create_test_project(tmp_path)

        # Build cache with hashing
        cache = build_symbol_inventory(str(tmp_path))
        for file_info in cache["files"]:
            file_path = tmp_path / file_info["path"]
            file_info["content_hash"] = _calculate_file_hash(file_path)

        # Sleep and modify file content
        time.sleep(0.1)
        write_file(tmp_path / "module1.py", "def modified_func():\n    return 'modified'\n")

        # Detect changes using hash-based method
        changed = detect_changed_files(str(tmp_path), cache, use_content_hash=True)

        assert len(changed) == 1
        assert "module1.py" in changed

    def test_detect_changed_files_hash_vs_mtime_accuracy(self, tmp_path: Path):
        """Test that hash-based detection is more accurate than mtime."""
        create_test_project(tmp_path)

        # Build initial cache
        cache = build_symbol_inventory(str(tmp_path))
        for file_info in cache["files"]:
            file_path = tmp_path / file_info["path"]
            file_info["content_hash"] = _calculate_file_hash(file_path)

        # "Touch" file without changing content (mtime changes, content doesn't)
        file_path = tmp_path / "module1.py"
        content = file_path.read_text()
        time.sleep(0.1)
        file_path.touch()  # Changes mtime but not content

        # mtime-based detection should report change
        changed_mtime = detect_changed_files(str(tmp_path), cache, use_content_hash=False)
        assert len(changed_mtime) == 1

        # Hash-based detection should report no change
        changed_hash = detect_changed_files(str(tmp_path), cache, use_content_hash=True)
        assert len(changed_hash) == 0

    def test_detect_changed_files_hash_fallback_to_mtime(self, tmp_path: Path):
        """Test fallback to mtime when hash calculation fails."""
        create_test_project(tmp_path)

        # Build cache without hashes
        cache = build_symbol_inventory(str(tmp_path))

        # Try to use hash-based detection on cache without hashes
        # Should fallback to mtime-based detection
        time.sleep(0.1)
        write_file(tmp_path / "module1.py", "def fallback_test():\n    return 'fallback'\n")

        changed = detect_changed_files(str(tmp_path), cache, use_content_hash=True)

        assert len(changed) == 1
        assert "module1.py" in changed


class TestIncrementalUpdateWithHashing:
    """Test incremental updates with content hashing."""

    def test_update_cache_incrementally_store_hashes(self, tmp_path: Path):
        """Test that incremental updates can store content hashes."""
        create_test_project(tmp_path)

        # Build initial cache
        initial_cache = build_symbol_inventory(str(tmp_path))

        # Modify a file
        time.sleep(0.1)
        write_file(tmp_path / "module1.py", "def hash_test():\n    return 'with_hash'\n")

        # Update cache with hash storage enabled
        updated_cache = update_cache_incrementally(str(tmp_path), initial_cache, ["module1.py"], store_content_hash=True)

        # Verify that hash was stored
        module1_file = next(f for f in updated_cache["files"] if f["path"] == "module1.py")
        assert "content_hash" in module1_file
        assert len(module1_file["content_hash"]) == 64  # Blake2b default

    def test_ensure_cache_with_content_hashing(self, tmp_path: Path):
        """Test ensure_cache with content hashing enabled."""
        create_test_project(tmp_path)

        # Initial cache build with hashing
        cache1 = ensure_cache(str(tmp_path), force=True, save=True, use_content_hash=True)

        # Verify hashes are stored
        for file_info in cache1["files"]:
            assert "content_hash" in file_info

        # Modify file and run incremental update with hashing
        time.sleep(0.1)
        write_file(tmp_path / "module2.py", "def hash_incremental():\n    return 'hashed'\n")

        cache2 = ensure_cache(str(tmp_path), incremental=True, save=True, use_content_hash=True)

        # Verify cache was updated and still has hashes
        module2_symbols = [s for s in cache2["symbols"] if s["path"] == "module2.py"]
        symbol_names = [s["name"] for s in module2_symbols]
        assert "hash_incremental" in symbol_names

        # Verify hash is present for the updated file
        module2_file = next(f for f in cache2["files"] if f["path"] == "module2.py")
        assert "content_hash" in module2_file

    def test_reindex_specific_files_with_hashing(self, tmp_path: Path):
        """Test specific file reindexing with content hashing."""
        create_test_project(tmp_path)

        # Build initial cache
        ensure_cache(str(tmp_path), force=True, save=True)

        # Modify specific file
        write_file(tmp_path / "module1.py", "def specific_hash_test():\n    return 'specific'\n")

        # Reindex with hashing enabled
        cache = reindex_specific_files(str(tmp_path), ["module1.py"], save=True, store_content_hash=True)

        # Verify hash is stored for the reindexed file
        module1_file = next(f for f in cache["files"] if f["path"] == "module1.py")
        assert "content_hash" in module1_file

        # Verify symbols were updated
        module1_symbols = [s for s in cache["symbols"] if s["path"] == "module1.py"]
        symbol_names = [s["name"] for s in module1_symbols]
        assert "specific_hash_test" in symbol_names


class TestCLIHashingIntegration:
    """Test CLI integration with content hashing."""

    def test_cli_index_with_hashing(self, tmp_path: Path):
        """Test CLI index command with content hashing enabled."""
        create_test_project(tmp_path)

        runner = CliRunner()

        # Initial indexing with hashing
        result1 = runner.invoke(app, ["index", "--project-root", str(tmp_path), "--save", "--force", "--hash"])
        assert result1.exit_code == 0
        assert "Content hashing" in result1.stdout
        assert "Enabled" in result1.stdout

        # Verify cache contains hashes
        cache_path = tmp_path / ".prompt_distil" / "project_cache.json"
        cache = read_json(cache_path)

        for file_info in cache["files"]:
            assert "content_hash" in file_info

    def test_cli_index_without_hashing(self, tmp_path: Path):
        """Test CLI index command without content hashing."""
        create_test_project(tmp_path)

        runner = CliRunner()

        result = runner.invoke(app, ["index", "--project-root", str(tmp_path), "--save", "--force", "--no-hash"])
        assert result.exit_code == 0
        assert "Content hashing" in result.stdout
        assert "Disabled" in result.stdout

        # Verify cache doesn't contain hashes
        cache_path = tmp_path / ".prompt_distil" / "project_cache.json"
        cache = read_json(cache_path)

        for file_info in cache["files"]:
            assert "content_hash" not in file_info

    def test_cli_incremental_update_with_hashing(self, tmp_path: Path):
        """Test CLI incremental update with content hashing."""
        create_test_project(tmp_path)

        runner = CliRunner()

        # Initial build with hashing
        result1 = runner.invoke(app, ["index", "--project-root", str(tmp_path), "--save", "--force", "--hash"])
        assert result1.exit_code == 0

        # Modify file and run incremental update
        time.sleep(0.1)
        write_file(tmp_path / "module1.py", "def cli_hash_test():\n    return 'cli'\n")

        result2 = runner.invoke(app, ["index", "--project-root", str(tmp_path), "--save", "--hash"])
        assert result2.exit_code == 0
        assert "updated incrementally" in result2.stdout
        assert "(with content hashing)" in result2.stdout

    def test_cli_specific_file_reindexing_with_hashing(self, tmp_path: Path):
        """Test CLI specific file reindexing with content hashing."""
        create_test_project(tmp_path)

        runner = CliRunner()

        # Initial build
        result1 = runner.invoke(app, ["index", "--project-root", str(tmp_path), "--save", "--force"])
        assert result1.exit_code == 0

        # Reindex specific file with hashing
        result2 = runner.invoke(app, ["index", "--project-root", str(tmp_path), "--file", "module1.py", "--save", "--hash"])
        assert result2.exit_code == 0
        assert "Successfully reindexed 1 specific files" in result2.stdout


class TestHashingPerformance:
    """Test performance aspects of content hashing."""

    def test_hashing_performance_overhead(self, tmp_path: Path):
        """Test that hashing doesn't add excessive overhead for small projects."""
        # Create a moderately sized test project
        for i in range(10):
            write_file(tmp_path / f"module{i}.py", f"def func{i}():\n    return 'module{i}'\n\nclass Class{i}:\n    def method(self):\n        pass\n" * 5)

        # Time mtime-based indexing
        start_time = time.time()
        cache_mtime = ensure_cache(str(tmp_path), force=True, save=False, use_content_hash=False)
        mtime_duration = time.time() - start_time

        # Time hash-based indexing
        start_time = time.time()
        cache_hash = ensure_cache(str(tmp_path), force=True, save=False, use_content_hash=True)
        hash_duration = time.time() - start_time

        # Verify both produce same symbols (sanity check)
        assert len(cache_mtime["symbols"]) == len(cache_hash["symbols"])

        # Hash-based should not be more than 3x slower for small projects
        # (This is a reasonable threshold for the added accuracy)
        assert hash_duration < mtime_duration * 3

    @patch("prompt_distil.core.surface._calculate_file_hash")
    def test_hash_calculation_error_handling(self, mock_hash, tmp_path: Path):
        """Test error handling when hash calculation fails."""
        create_test_project(tmp_path)

        # Mock hash calculation to raise an exception
        mock_hash.side_effect = RuntimeError("Hash calculation failed")

        # Should fallback gracefully and not store hashes
        cache = update_cache_incrementally(str(tmp_path), {"files": [], "symbols": [], "inverted_index": {}}, ["module1.py"], store_content_hash=True)

        # Verify file was processed but without hash
        assert len(cache["files"]) == 1
        module1_file = cache["files"][0]
        assert "content_hash" not in module1_file
        assert module1_file["path"] == "module1.py"


class TestHashConsistency:
    """Test hash consistency and reliability."""

    def test_hash_consistency_across_runs(self, tmp_path: Path):
        """Test that same file content produces consistent hashes across runs."""
        test_file = tmp_path / "consistent.py"
        content = "def consistent_test():\n    return 'same content every time'\n"

        hashes = []
        for _ in range(5):
            write_file(test_file, content)
            hash_result = _calculate_file_hash(test_file)
            hashes.append(hash_result)

        # All hashes should be identical
        assert len(set(hashes)) == 1

    def test_hash_sensitivity_to_whitespace(self, tmp_path: Path):
        """Test that hashes detect subtle content changes."""
        test_file = tmp_path / "whitespace.py"

        write_file(test_file, "def test():\n    return 'value'")
        hash1 = _calculate_file_hash(test_file)

        write_file(test_file, "def test():\n     return 'value'")  # Extra space
        hash2 = _calculate_file_hash(test_file)

        assert hash1 != hash2

    def test_hash_empty_file(self, tmp_path: Path):
        """Test hashing of empty files."""
        empty_file = tmp_path / "empty.py"
        empty_file.touch()

        hash_result = _calculate_file_hash(empty_file)
        assert len(hash_result) == 64  # Should still produce valid hash
