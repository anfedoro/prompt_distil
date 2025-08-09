"""
Configuration management for the Prompt Distiller.

This module handles environment variables, API keys, and model configurations
using python-dotenv for explicit, project-scoped .env loading. No implicit loading occurs at import time.
"""

import os
import shutil
from functools import lru_cache
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI


class ConfigError(Exception):
    """Raised when configuration is invalid or missing."""

    pass


@lru_cache(maxsize=32)
def load_config(env_path: Optional[str] = None, override: bool = False) -> None:
    """Explicitly load environment variables from the given .env file path.

    Notes:
    - This function does NOT perform implicit loading when env_path is None.
    - Callers should pass a project-scoped env path resolved via helpers in this module.
    """
    if env_path:
        load_dotenv(dotenv_path=env_path, override=override)


class Config:
    """Configuration settings for the Prompt Distiller."""

    def __init__(self):
        # Do not implicitly load any .env here. Consumers must call project-scoped loaders explicitly.
        pass

    @property
    def openai_api_key(self) -> str:
        """Get OpenAI API key from environment."""
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise ConfigError("OPENAI_API_KEY not found in environment. Please set it in your environment or .env file.")
        return key

    @property
    def llm_model(self) -> str:
        """Get the LLM model name for both distillation and reconciliation (default: gpt-4o-mini)."""
        return os.getenv("LLM_MODEL", os.getenv("DISTIL_MODEL", "gpt-4o-mini"))

    @property
    def distil_model(self) -> str:
        """Get the model name for distillation (deprecated, use llm_model)."""
        return self.llm_model

    @property
    def is_reasoning_model(self) -> bool:
        """Check if the configured model is a reasoning model (default: False)."""
        value = os.getenv("IS_REASONING_MODEL", "false").lower()
        return value in ("true", "1", "yes", "on")

    @property
    def asr_model(self) -> str:
        """Get the model name for ASR (default: whisper-1)."""
        return os.getenv("ASR_MODEL", "whisper-1")

    @property
    def openai_timeout(self) -> int:
        """Get OpenAI API timeout in seconds (default: 60)."""
        return int(os.getenv("OPENAI_TIMEOUT", "60"))

    @property
    def max_retries(self) -> int:
        """Get maximum number of retries for API calls (default: 3)."""
        return int(os.getenv("MAX_RETRIES", "3"))


# Global config instance
config = Config()

# --- Project-scoped environment helpers ---

DEFAULT_ENV_FILENAME = os.getenv("PD_ENV_FILENAME", ".env")
ENV_FILE_ENV_VARS = ("PD_ENV_FILE", "PROMPT_DISTIL_ENV_FILE")
PROJECT_ROOT_ENV_VARS = ("PD_PROJECT_ROOT", "PROMPT_DISTIL_PROJECT_ROOT")


def detect_project_root(start_dir: Optional[str] = None) -> Optional[Path]:
    """Detect project root by looking for a .prompt_distil directory upwards from start_dir (or CWD)."""
    start = Path(start_dir) if start_dir else Path.cwd()
    for current in [start] + list(start.parents):
        if (current / ".prompt_distil").exists():
            return current
    return None


def get_project_metadata_dir(project_root: Optional[str] = None) -> Path:
    """Return the .prompt_distil directory for a given or detected project root."""
    root = Path(project_root) if project_root else (detect_project_root() or Path.cwd())
    return root / ".prompt_distil"


def ensure_project_metadata_dir(project_root: Optional[str] = None) -> Path:
    """
    Ensure the .prompt_distil directory exists for the project and return its path.
    This function is safe for indexing: it does not read or load any environment files.
    """
    meta_dir = get_project_metadata_dir(project_root)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return meta_dir


def get_project_env_path(project_root: Optional[str] = None, filename: str = DEFAULT_ENV_FILENAME) -> Path:
    """
    Compute the path to the project-scoped environment file inside .prompt_distil.
    Falls back to '.env' inside .prompt_distil if the preferred filename is missing.
    """
    meta = get_project_metadata_dir(project_root)
    preferred = meta / filename
    return preferred


def ensure_project_env(
    project_root: Optional[str] = None,
    source_env: Optional[str] = None,
    filename: str = DEFAULT_ENV_FILENAME,
    overwrite: bool = False,
) -> Path:
    """
    Create or copy a project-scoped env file under .prompt_distil.
    This is indexing-neutral: it NEVER loads env values, it only writes/places the file.

    Behavior:
    - If target exists and overwrite is False, the existing file is preserved.
    - If source_env is provided and exists, it's copied to the target.
    - Else if <project_root>/.env exists, it's copied to the target.
    - Else, a minimal template is created at the target.

    Returns:
        Path to the env file under the project's .prompt_distil directory.
    """
    meta = ensure_project_metadata_dir(project_root)
    target = meta / filename
    legacy = meta / "configuration.env"
    if (not target.exists()) and legacy.exists():
        shutil.copyfile(legacy, target)
        return target
    if target.exists() and not overwrite:
        return target

    # Determine source candidates
    src_candidates = []
    if source_env:
        src_candidates.append(Path(source_env))
    root = Path(project_root) if project_root else Path.cwd()
    src_candidates.append(root / ".env")

    for candidate in src_candidates:
        if candidate.is_file():
            shutil.copyfile(candidate, target)
            return target

    # No source found: create a minimal template
    template = (
        "# Project-scoped environment for prompt_distil\n"
        "# Edit values as needed. This file is not loaded during indexing.\n"
        "LLM_MODEL=gpt-4o-mini\n"
        "IS_REASONING_MODEL=false\n"
        "MODEL_TEMPERATURE=0.2\n"
        "OPENAI_TIMEOUT=60\n"
        "MAX_RETRIES=3\n"
        "# OPENAI_API_KEY=your-key-here\n"
    )
    target.write_text(template, encoding="utf-8")
    return target


def describe_project_env_location(project_root: Optional[str] = None, filename: str = DEFAULT_ENV_FILENAME) -> str:
    """
    Return a human-readable message describing the project-scoped env file location.
    Useful for reporting after indexing completes.
    """
    path = get_project_env_path(project_root, filename)
    exists = path.exists()
    return f"{path} ({'exists' if exists else 'will be created on demand'})"


def load_project_env(project_root: Optional[str] = None, filename: str = DEFAULT_ENV_FILENAME, override: bool = False) -> Optional[str]:
    """
    Load a project-scoped environment file, if available.

    Load order (first match wins):
    1) Explicit env file path via PD_ENV_FILE or PROMPT_DISTIL_ENV_FILE
    2) <project_root>/.prompt_distil/<filename> (default: .env)
    3) LEGACY fallback: <project_root>/.prompt_distil/configuration.env (will be migrated to .env)

    Returns the path loaded, or None if nothing was loaded.
    """
    # 1) Explicit env file via environment variable
    for var in ENV_FILE_ENV_VARS:
        explicit = os.getenv(var)
        if explicit and Path(explicit).is_file():
            load_config(explicit, override=override)
            return explicit

    # 2) Resolve project_root from env or detection
    if project_root is None:
        for var in PROJECT_ROOT_ENV_VARS:
            if os.getenv(var):
                project_root = os.getenv(var)
                break
    if project_root is None:
        detected = detect_project_root()
        project_root = str(detected) if detected else None

    # 3) Attempt project-scoped files
    if project_root:
        env_path = get_project_env_path(project_root, filename)
        if env_path.exists():
            load_config(str(env_path), override=override)
            return str(env_path)
        # Legacy fallback: configuration.env → migrate to .env and load
        legacy_path = get_project_metadata_dir(project_root) / "configuration.env"
        if legacy_path.exists():
            try:
                if not env_path.exists():
                    import shutil

                    shutil.copyfile(legacy_path, env_path)
            except Exception:
                # Migration best-effort; continue to load legacy file even if copy fails
                pass
            load_config(str(legacy_path), override=override)
            return str(legacy_path)

    return None


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """
    Get configured OpenAI client with timeout and retry settings.

    Behavior:
    - Does not implicitly load a .env from the current working directory.
    - If OPENAI_API_KEY is missing, attempts to load a project-scoped env:
      PD_ENV_FILE → <project>/.prompt_distil/.env

    Returns:
        OpenAI client instance

    Raises:
        ConfigError: If API key is not configured after project env lookup
    """
    # Ensure API key is available; try project-scoped env as a fallback
    if not os.getenv("OPENAI_API_KEY"):
        loaded_path = load_project_env()
        if not os.getenv("OPENAI_API_KEY"):
            where = loaded_path or f".prompt_distil/{DEFAULT_ENV_FILENAME} (or configuration.env)"
            raise ConfigError(
                f"OPENAI_API_KEY not found in environment. Looked for project env at {where}. "
                f"Set it via environment, PD_ENV_FILE, or place it under .prompt_distil/{DEFAULT_ENV_FILENAME} (or configuration.env)."
            )
    try:
        return OpenAI(
            api_key=config.openai_api_key,
            timeout=config.openai_timeout,
            max_retries=config.max_retries,
        )
    except Exception as e:
        raise ConfigError(f"Failed to create OpenAI client: {e}")


def validate_config() -> None:
    """
    Validate that all required configuration is present.

    Raises:
        ConfigError: If required configuration is missing
    """
    # Creation of the client will validate presence of the API key,
    # attempting a project-scoped env load if necessary.
    _ = get_client()
