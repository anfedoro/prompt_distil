import json
import os
from pathlib import Path
from typing import Optional

import pytest
from typer.testing import CliRunner

from prompt_distil.core.config import (
    ensure_project_env,
    get_client,
    get_project_env_path,
    load_project_env,
)
from prompt_distil.main import app


class DummyOpenAI:
    """
    Minimal dummy stand-in for OpenAI client to avoid real network/API calls.

    Captures initialization parameters so tests can assert which API key was used.
    """

    def __init__(self, api_key: Optional[str] = None, timeout: Optional[int] = None, max_retries: Optional[int] = None, **_: object):
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries


def write_file(path: Path, content: str) -> None:
    """Helper to write a text file with UTF-8 encoding."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def read_json(path: Path) -> dict:
    """Helper to read a JSON file."""
    return json.loads(path.read_text(encoding="utf-8"))


def list_contains_path(files: list[dict], rel_path: str) -> bool:
    """Check if a list of file info dicts contains a specific relative path."""
    for item in files:
        # file info is stored as {"path": "...", ...}
        if isinstance(item, dict) and item.get("path") == rel_path:
            return True
    return False


@pytest.fixture(autouse=True)
def clear_get_client_cache():
    """
    Ensure get_client cache is cleared before and after tests that may rely on it.
    """
    try:
        get_client.cache_clear()
    except Exception:
        pass
    yield
    try:
        get_client.cache_clear()
    except Exception:
        pass


def test_index_ignores_init_and_creates_project_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verify that:
    - __init__.py files are ignored during indexing (no file record, no symbols from it).
    - Indexing does not load any .env into process environment.
    - A project-scoped env file is created in .prompt_distil/.env.
    """
    project_root = tmp_path / "proj"
    pkg_dir = project_root / "pkg"
    pkg_dir.mkdir(parents=True)

    # Create a __init__.py that would otherwise be indexable
    write_file(pkg_dir / "__init__.py", "def init_func():\n    return 'should_not_be_indexed'\n")

    # Create a real module file
    write_file(pkg_dir / "module.py", "def real_func():\n    return 'ok'\n")

    # Also create a project root .env to ensure it's not loaded during indexing
    write_file(project_root / ".env", "OPENAI_API_KEY=should_not_be_loaded\n")

    # Ensure test environment is clean
    monkeypatch.setenv("PYTHONHASHSEED", "0")  # stability
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Run the Typer CLI 'index' command
    runner = CliRunner()
    result = runner.invoke(app, ["index", "--project-root", str(project_root), "--save"])

    assert result.exit_code == 0, f"CLI index failed: {result.stdout}\n{result.stderr}"

    # Ensure no OPENAI_API_KEY leaked into process env during indexing
    assert os.getenv("OPENAI_API_KEY") is None

    # Verify cache file exists
    cache_path = project_root / ".prompt_distil" / "project_cache.json"
    assert cache_path.exists(), "Expected project cache to be saved under .prompt_distil/project_cache.json"

    cache = read_json(cache_path)

    # Files list should not include pkg/__init__.py
    assert not list_contains_path(cache.get("files", []), "pkg/__init__.py"), "__init__.py must be excluded from files list"

    # Symbols should not come from __init__.py
    for sym in cache.get("symbols", []):
        assert sym.get("path") != "pkg/__init__.py", "Symbols from __init__.py must not be indexed"

    # Project-scoped env should exist
    env_path = project_root / ".prompt_distil" / ".env"
    assert env_path.exists(), "Project-scoped env (.env) should be created during indexing"


def test_load_project_env_prefers_project_scoped_over_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verify that load_project_env loads the project-scoped env file, not the CWD .env.
    Also verify get_client uses that environment (with OpenAI patched to DummyOpenAI).
    """
    # Create two separate directories: project and an unrelated working directory
    project_root = tmp_path / "proj2"
    other_cwd = tmp_path / "cwd"
    project_root.mkdir()
    other_cwd.mkdir()

    # Create CWD .env with a different key (should not be used)
    write_file(other_cwd / ".env", "OPENAI_API_KEY=cwd-key\n")

    # Create project-scoped env under .prompt_distil with the intended key
    meta_dir = project_root / ".prompt_distil"
    meta_dir.mkdir(parents=True, exist_ok=True)
    write_file(meta_dir / ".env", "OPENAI_API_KEY=project-key\n")

    # Start with a clean environment and change CWD to directory with .env
    monkeypatch.chdir(other_cwd)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    # Load project-scoped env explicitly
    loaded = load_project_env(project_root=str(project_root))
    assert loaded is not None, "Project-scoped env should be found and loaded"
    assert os.getenv("OPENAI_API_KEY") == "project-key", "Project-scoped env must take precedence over CWD .env"

    # Patch OpenAI to our dummy to avoid real network and capture parameters
    import prompt_distil.core.config as config_mod

    monkeypatch.setattr(config_mod, "OpenAI", DummyOpenAI, raising=True)

    # Ensure cached client is reset and then fetch a client
    get_client.cache_clear()
    client = get_client()
    assert isinstance(client, DummyOpenAI)
    assert client.api_key == "project-key", "Client must be initialized with API key from project-scoped env"


def test_ensure_project_env_copies_or_templates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """
    Verify ensure_project_env behavior:
    - If project root .env exists, it is copied to .prompt_distil/.env
    - Otherwise, a minimal template is created.
    """
    # Case 1: .env exists at root and should be copied
    project_with_env = tmp_path / "proj_with_env"
    project_with_env.mkdir()
    write_file(project_with_env / ".env", "LLM_MODEL=foo\nOPENAI_API_KEY=xyz\n")

    path1 = ensure_project_env(str(project_with_env))
    assert path1.exists()
    content1 = path1.read_text(encoding="utf-8")
    assert "LLM_MODEL=foo" in content1
    assert "OPENAI_API_KEY=xyz" in content1

    # Case 2: No .env exists, ensure template is created
    project_no_env = tmp_path / "proj_no_env"
    project_no_env.mkdir()

    path2 = ensure_project_env(str(project_no_env))
    assert path2.exists()
    content2 = path2.read_text(encoding="utf-8")
    assert "Project-scoped environment for prompt_distil" in content2
    assert "LLM_MODEL" in content2
    assert "MODEL_TEMPERATURE" in content2

    # Validate helper get_project_env_path resolves to the same path
    resolved = get_project_env_path(str(project_no_env))
    assert resolved == path2
