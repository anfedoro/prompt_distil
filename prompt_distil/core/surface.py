"""
Surface-level project indexing and search functionality.

This module provides utilities for listing files, searching project content,
and reading files without deep AST analysis. It uses only stdlib functionality
for lightweight project exploration.

Includes symbol inventory caching for fast symbol lookup and reconciliation.
"""

import ast
import datetime
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


def build_symbol_inventory(root: str = ".", globs: Optional[List[str]] = None, max_files: int = 1000, max_bytes: int = 200_000) -> Dict:
    """
    Build a symbol inventory cache from project files using AST parsing.

    Args:
        root: Root directory to search (default: ".")
        globs: List of glob patterns to search (default: ["**/*.py"])
        max_files: Maximum number of files to process
        max_bytes: Maximum bytes per file to process

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

    # Collect files matching globs
    all_files = []
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


def ensure_cache(root: str, globs: Optional[List[str]] = None, force: bool = False, save: bool = True) -> Dict:
    """
    Load cache if present; otherwise build and optionally save.

    Args:
        root: Root directory for cache operations
        globs: Glob patterns to use for building cache
        force: Force rebuild even if cache exists
        save: Whether to save the cache after building

    Returns:
        Symbol cache dictionary
    """
    if not force:
        cache = load_cache(root)
        if cache is not None:
            return cache

    # Build new cache
    cache = build_symbol_inventory(root, globs)

    if save:
        save_cache(root, cache)

    return cache
