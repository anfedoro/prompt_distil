"""
Surface-level project indexing and search functionality.

This module provides utilities for listing files, searching project content,
and reading files without deep AST analysis. It uses only stdlib functionality
for lightweight project exploration.

Includes symbol inventory caching for fast symbol lookup and reconciliation.
"""

import ast
import datetime
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from .progress import reporter


class SurfaceError(Exception):
    """Raised when surface operations fail."""

    pass


class ProjectSurface:
    """
    Provides surface-level project indexing and search capabilities.

    This class offers file listing, content search, and file reading
    without requiring complex parsing or AST analysis.
    """

    def __init__(self, project_root: Optional[str] = None):
        """
        Initialize project surface with optional root directory.

        Args:
            project_root: Root directory for project operations.
                         If None, uses current working directory.
        """
        self.project_root = Path(project_root) if project_root else Path.cwd()
        if not self.project_root.exists():
            raise SurfaceError(f"Project root does not exist: {self.project_root}")

    def list_files(self, pattern: Optional[str] = None) -> List[str]:
        """
        List files in the project matching an optional glob pattern.

        Args:
            pattern: Optional glob pattern (e.g., "**/*.py", "src/**/*.js")
                    If None, lists all files recursively

        Returns:
            List of relative file paths
        """
        if pattern is None:
            pattern = "**/*"

        try:
            # Use pathlib for more reliable cross-platform globbing
            matches = list(self.project_root.glob(pattern))

            # Filter out directories and return relative paths
            files = []
            for match in matches:
                if match.is_file():
                    relative_path = match.relative_to(self.project_root)
                    files.append(str(relative_path))

            return sorted(files)

        except Exception as e:
            raise SurfaceError(f"Failed to list files with pattern '{pattern}': {e}")

    def search_project(self, query: str, top_k: int = 5, file_pattern: Optional[str] = None) -> List[Dict]:
        """
        Search for text content across project files.

        Args:
            query: Search query (treated as regex pattern)
            top_k: Maximum number of results to return
            file_pattern: Optional glob pattern to limit search to specific files

        Returns:
            List of dictionaries with keys: 'path', 'line', 'line_number', 'snippet'
        """
        results = []

        # Get files to search
        files_to_search = self.list_files(file_pattern)

        # Common file types to exclude from search
        excluded_extensions = {".pyc", ".pyo", ".so", ".dll", ".exe", ".bin", ".jpg", ".jpeg", ".png", ".gif", ".pdf", ".zip"}

        try:
            # Compile regex pattern
            pattern = re.compile(query, re.IGNORECASE)
        except re.error as e:
            raise SurfaceError(f"Invalid regex pattern '{query}': {e}")

        for file_path in files_to_search:
            # Skip binary and unwanted files
            if any(file_path.lower().endswith(ext) for ext in excluded_extensions):
                continue

            # Skip hidden files and common build directories
            path_parts = Path(file_path).parts
            if any(part.startswith(".") or part in {"__pycache__", "node_modules", "build", "dist"} for part in path_parts):
                continue

            try:
                full_path = self.project_root / file_path

                # Try to read as text file
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        lines = f.readlines()
                except UnicodeDecodeError:
                    # Skip binary files that can't be decoded
                    continue

                # Search each line
                for line_num, line in enumerate(lines, 1):
                    if pattern.search(line):
                        # Create snippet with some context
                        snippet_lines = []
                        start_line = max(0, line_num - 2)
                        end_line = min(len(lines), line_num + 2)

                        for i in range(start_line, end_line):
                            prefix = ">>> " if i == line_num - 1 else "    "
                            snippet_lines.append(f"{prefix}{lines[i].rstrip()}")

                        results.append({"path": file_path, "line": line.strip(), "line_number": line_num, "snippet": "\n".join(snippet_lines)})

                        # Stop if we have enough results
                        if len(results) >= top_k:
                            return results

            except Exception:
                # Skip files that can't be read
                continue

        return results

    def get_project_structure(self, max_depth: int = 3) -> Dict:
        """
        Get a high-level overview of project structure.

        Args:
            max_depth: Maximum directory depth to explore

        Returns:
            Dictionary representing project structure
        """
        structure = {"root": str(self.project_root), "directories": [], "files": [], "file_types": {}}

        try:
            for root, dirs, files in os.walk(self.project_root):
                root_path = Path(root)
                relative_root = root_path.relative_to(self.project_root)

                # Skip if too deep
                if len(relative_root.parts) >= max_depth:
                    dirs.clear()  # Don't recurse deeper
                    continue

                # Skip hidden and build directories
                dirs[:] = [d for d in dirs if not d.startswith(".") and d not in {"__pycache__", "node_modules", "build", "dist", ".git"}]

                # Add directory info
                if str(relative_root) != ".":
                    structure["directories"].append(str(relative_root))

                # Process files
                for file in files:
                    if file.startswith("."):
                        continue

                    file_path = relative_root / file
                    structure["files"].append(str(file_path))

                    # Track file types
                    extension = Path(file).suffix.lower()
                    if extension:
                        structure["file_types"][extension] = structure["file_types"].get(extension, 0) + 1

        except Exception as e:
            raise SurfaceError(f"Failed to analyze project structure: {e}")

        return structure


# Convenience functions for simple use cases


def build_symbol_inventory(
    root: str = ".",
    globs: Optional[List[str]] = None,
    max_files: int = 1000,
    max_bytes: int = 200_000,
    target_files: Optional[List[str]] = None,
    store_content_hash: bool = False,
) -> Dict:
    """
    Build a symbol inventory cache from project files using AST parsing.

    Args:
        root: Root directory to search (default: ".")
        globs: List of glob patterns to search (default: ["**/*.py"])
        max_files: Maximum number of files to process
        max_bytes: Maximum bytes per file to process
        target_files: Optional list of specific files to process (for incremental updates)
        store_content_hash: Store content hashes for better change detection

    Returns:
        Dictionary with cache structure containing symbols and metadata
    """
    if globs is None:
        globs = ["**/*.py"]

    root_path = Path(root).absolute()

    reporter.step("Scanning project files…")

    # Initialize cache structure
    cache = {
        "version": 1,
        "generated_at": datetime.datetime.now().isoformat(),
        "root": str(root_path.absolute()),
        "globs": globs,
        "files": [],
        "symbols": [],
        "inverted_index": {},
    }

    files_processed = 0

    # Collect files matching globs or use target files for incremental updates
    all_files = []
    if target_files:
        # Process only specific target files
        for target_file in target_files:
            file_path = root_path / target_file
            if file_path.exists() and file_path.is_file() and _should_include_file(file_path):
                all_files.append(file_path)
    else:
        # Process all files matching globs
        for glob_pattern in globs:
            for file_path in root_path.glob(glob_pattern):
                if file_path.is_file() and _should_include_file(file_path):
                    all_files.append(file_path)

    # Sort by size to process smaller files first
    all_files.sort(key=lambda f: f.stat().st_size)

    reporter.step("Extracting symbols from files…")

    for file_path in all_files:
        if files_processed >= max_files:
            break

        try:
            rel_path = file_path.relative_to(root_path)
            stat = file_path.stat()

            # Skip files that are too large
            if stat.st_size > max_bytes:
                continue

            # Add file metadata
            file_info = {"path": str(rel_path), "mtime": stat.st_mtime, "size": stat.st_size}

            # Add content hash if requested
            if store_content_hash:
                try:
                    file_info["content_hash"] = _calculate_file_hash(file_path)
                except Exception:
                    # If hashing fails, continue without hash
                    pass

            cache["files"].append(file_info)

            # Extract symbols if it's a Python file
            if file_path.suffix == ".py":
                symbols = _extract_python_symbols(file_path, str(rel_path))
                cache["symbols"].extend(symbols)

            files_processed += 1

        except Exception:
            # Skip files that can't be processed
            continue

    reporter.step("Building symbol index…")

    # Build inverted index
    for symbol in cache["symbols"]:
        name = symbol["name"]
        location = f"{symbol['path']}#L{symbol['lineno']}"

        if name not in cache["inverted_index"]:
            cache["inverted_index"][name] = []
        cache["inverted_index"][name].append(location)

    return cache


def _should_include_file(file_path: Path) -> bool:
    """Check if file should be included in symbol inventory."""
    # Skip hidden files and directories
    parts = file_path.parts
    for part in parts:
        if part.startswith(".") and part not in {".", ".."}:
            return False

    # Skip common build/dependency directories
    exclude_dirs = {"node_modules", "dist", "build", ".git", "__pycache__", ".venv"}
    for part in parts:
        if part in exclude_dirs:
            return False

    # Skip Python package initializers to avoid indexing package roots
    if file_path.name == "__init__.py":
        return False

    return True


def _extract_python_symbols(file_path: Path, rel_path: str) -> List[Dict]:
    """Extract symbols from Python file using AST."""
    symbols = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        tree = ast.parse(content)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbols.append({"name": node.name, "kind": "function", "path": rel_path, "lineno": node.lineno})
            elif isinstance(node, ast.ClassDef):
                symbols.append({"name": node.name, "kind": "class", "path": rel_path, "lineno": node.lineno})

    except Exception:
        # Skip files that can't be parsed
        pass

    return symbols


def save_cache(root: str, cache: Dict) -> None:
    """Save symbol cache to JSON file under root/.prompt_distil/."""
    cache_path = Path(root) / ".prompt_distil" / "project_cache.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def load_cache(root: str) -> Optional[Dict]:
    """Load symbol cache from JSON file under root/.prompt_distil/."""
    cache_path = Path(root) / ".prompt_distil" / "project_cache.json"

    if not cache_path.exists():
        return None

    try:
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def detect_changed_files(root: str, cache: Dict, globs: Optional[List[str]] = None, use_content_hash: bool = False) -> List[str]:
    """
    Detect files that have changed since last indexing by comparing modification times.

    Args:
        root: Root directory for operations
        cache: Existing cache with file metadata
        globs: Glob patterns to check (default: ["**/*.py"])
        use_content_hash: Use content hashing for more accurate change detection

    Returns:
        List of relative paths of files that need reindexing
    """
    if globs is None:
        globs = ["**/*.py"]

    root_path = Path(root).absolute()
    changed_files = []

    # Create a lookup of cached file info by path
    cached_files = {file_info["path"]: file_info for file_info in cache.get("files", [])}

    # Check all files matching globs
    for glob_pattern in globs:
        for file_path in root_path.glob(glob_pattern):
            if file_path.is_file() and _should_include_file(file_path):
                rel_path = str(file_path.relative_to(root_path))

                try:
                    current_mtime = file_path.stat().st_mtime
                    cached_file_info = cached_files.get(rel_path)

                    # Check if file is new
                    if cached_file_info is None:
                        changed_files.append(rel_path)
                        continue

                    # Use content hashing if requested and available
                    if use_content_hash and "content_hash" in cached_file_info:
                        try:
                            current_hash = _calculate_file_hash(file_path)
                            if current_hash != cached_file_info["content_hash"]:
                                changed_files.append(rel_path)
                        except Exception:
                            # Fallback to mtime if hashing fails
                            if current_mtime > cached_file_info["mtime"]:
                                changed_files.append(rel_path)
                    else:
                        # Use mtime-based detection
                        if current_mtime > cached_file_info["mtime"]:
                            changed_files.append(rel_path)
                except Exception:
                    # If we can't check the file, consider it changed
                    changed_files.append(rel_path)

    return changed_files


def update_cache_incrementally(root: str, cache: Dict, changed_files: List[str], max_bytes: int = 200_000, store_content_hash: bool = False) -> Dict:
    """
    Update cache by reindexing only changed files.

    Args:
        root: Root directory for operations
        cache: Existing cache to update
        changed_files: List of files that need reindexing
        max_bytes: Maximum bytes per file to process
        store_content_hash: Store content hashes in cache for better change detection

    Returns:
        Updated cache dictionary
    """
    if not changed_files:
        return cache

    root_path = Path(root).absolute()

    reporter.step(f"Reindexing {len(changed_files)} changed files…")

    # Remove old symbols and file info for changed files
    cache["files"] = [f for f in cache.get("files", []) if f["path"] not in changed_files]
    cache["symbols"] = [s for s in cache.get("symbols", []) if s["path"] not in changed_files]

    # Process changed files
    for file_path_str in changed_files:
        try:
            file_path = root_path / file_path_str
            stat = file_path.stat()

            # Skip files that are too large
            if stat.st_size > max_bytes:
                continue

            # Add updated file metadata
            file_info = {"path": file_path_str, "mtime": stat.st_mtime, "size": stat.st_size}

            # Add content hash if requested
            if store_content_hash:
                try:
                    file_info["content_hash"] = _calculate_file_hash(file_path)
                except Exception:
                    # If hashing fails, continue without hash
                    pass

            cache["files"].append(file_info)

            # Extract symbols if it's a Python file
            if file_path.suffix == ".py":
                symbols = _extract_python_symbols(file_path, file_path_str)
                cache["symbols"].extend(symbols)

        except Exception:
            # Skip files that can't be processed
            continue

    # Rebuild inverted index
    reporter.step("Rebuilding symbol index…")
    cache["inverted_index"] = {}
    for symbol in cache["symbols"]:
        name = symbol["name"]
        location = f"{symbol['path']}#L{symbol['lineno']}"

        if name not in cache["inverted_index"]:
            cache["inverted_index"][name] = []
        cache["inverted_index"][name].append(location)

    # Update metadata
    cache["generated_at"] = datetime.datetime.now().isoformat()

    return cache


def reindex_specific_files(root: str, files: List[str], save: bool = True, store_content_hash: bool = False) -> Dict:
    """
    Reindex specific files, updating the existing cache or creating a new one.

    Args:
        root: Root directory for operations
        files: List of specific files to reindex
        save: Whether to save the updated cache
        store_content_hash: Store content hashes for better change detection

    Returns:
        Updated cache dictionary
    """
    # Load existing cache or create new one
    cache = load_cache(root)
    if cache is None:
        # Create minimal cache structure
        cache = {
            "version": 1,
            "generated_at": datetime.datetime.now().isoformat(),
            "root": str(Path(root).absolute()),
            "globs": ["**/*.py"],
            "files": [],
            "symbols": [],
            "inverted_index": {},
        }

    # Update cache with specific files
    cache = update_cache_incrementally(root, cache, files, store_content_hash=store_content_hash)

    if save:
        save_cache(root, cache)

    return cache


def _calculate_file_hash(file_path: Path, algorithm: str = "blake2b") -> str:
    """
    Calculate hash of file content for change detection.

    Args:
        file_path: Path to the file to hash
        algorithm: Hash algorithm to use ('blake2b', 'sha256', 'md5')

    Returns:
        Hexadecimal hash string
    """
    if algorithm == "blake2b":
        hasher = hashlib.blake2b(digest_size=32)
    elif algorithm == "sha256":
        hasher = hashlib.sha256()
    elif algorithm == "md5":
        hasher = hashlib.md5()
    else:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    try:
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files efficiently
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    except Exception as e:
        raise RuntimeError(f"Failed to calculate hash for {file_path}: {e}")


def ensure_cache(
    root: str, globs: Optional[List[str]] = None, force: bool = False, save: bool = True, incremental: bool = True, use_content_hash: bool = False
) -> Dict:
    """
    Load cache if present; otherwise build and optionally save.
    Supports incremental updates by default.

    Args:
        root: Root directory for cache operations
        globs: Glob patterns to use for building cache
        force: Force full rebuild even if cache exists
        save: Whether to save the cache after building
        incremental: Use incremental updates when possible
        use_content_hash: Use content hashing for more accurate change detection

    Returns:
        Symbol cache dictionary
    """
    if force:
        # Force full rebuild
        cache = build_symbol_inventory(root, globs, store_content_hash=use_content_hash)
        if save:
            save_cache(root, cache)
        return cache

    # Try to load existing cache
    cache = load_cache(root)

    if cache is None:
        # No cache exists, build new one
        cache = build_symbol_inventory(root, globs, store_content_hash=use_content_hash)
        if save:
            save_cache(root, cache)
        return cache

    if not incremental:
        # User disabled incremental updates, return existing cache as-is
        return cache

    # Check for changed files and update incrementally
    changed_files = detect_changed_files(root, cache, globs, use_content_hash=use_content_hash)

    if changed_files:
        reporter.step(f"Found {len(changed_files)} changed files, updating cache incrementally…")
        cache = update_cache_incrementally(root, cache, changed_files, store_content_hash=use_content_hash)
        if save:
            save_cache(root, cache)
    else:
        reporter.step("No files changed since last indexing")

    return cache
