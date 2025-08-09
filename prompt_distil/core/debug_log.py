"""
Debug logging module for detailed reconcile_text hybrid mode analysis.

This module provides comprehensive logging for LLM requests, responses, and n-gram
comparisons during the reconcile_text process. Logs are stored in a dedicated
subfolder near the prompt_distil storage location.
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class DebugLogger:
    """
    Handles detailed debug logging for reconcile_text hybrid mode operations.

    Logs are stored in {project_root}/.prompt_distil/debug/ directory with
    timestamps and session identifiers for easy tracking.
    """

    def __init__(self, project_root: str = ".", enabled: bool = None):
        """
        Initialize debug logger.

        Args:
            project_root: Project root directory for log storage
            enabled: Override debug enable flag, uses PD_DEBUG env var if None
        """
        self.project_root = project_root
        # Check PD_DEBUG first, fallback to PD_DEBUG_RECONCILE for backward compatibility
        debug_enabled = os.getenv("PD_DEBUG", os.getenv("PD_DEBUG_RECONCILE", "0")) == "1"
        self.enabled = enabled if enabled is not None else debug_enabled
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        if self.enabled:
            self._setup_log_directory()

    def _setup_log_directory(self) -> None:
        """Create debug log directory structure."""
        self.log_dir = Path(self.project_root) / ".prompt_distil" / "debug"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create session-specific subdirectory
        self.session_dir = self.log_dir / f"session_{self.session_id}"
        self.session_dir.mkdir(exist_ok=True)

    def is_enabled(self) -> bool:
        """Check if debug logging is enabled."""
        return self.enabled

    def log_llm_request(self, prompt: str, symbols: List[str]) -> None:
        """
        Log LLM request details.

        Args:
            prompt: Full prompt sent to LLM
            symbols: List of candidate symbols provided
        """
        if not self.enabled:
            return

        timestamp = datetime.now().isoformat()
        log_data = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "step": "llm_request",
            "type": "request",
            "prompt": prompt,
            "candidate_symbols": symbols,
            "symbols_count": len(symbols),
        }

        filename = f"llm_request_{timestamp.replace(':', '-').replace('.', '_')}.json"
        log_file = self.session_dir / filename

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    def log_llm_response(self, response_content: str, original_text: str) -> None:
        """
        Log LLM response details.

        Args:
            response_content: Raw response from LLM
            original_text: Original input text for comparison
        """
        if not self.enabled:
            return

        timestamp = datetime.now().isoformat()
        log_data = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "step": "llm_response",
            "type": "response",
            "response_content": response_content,
            "original_text": original_text,
            "response_length": len(response_content),
            "original_length": len(original_text),
        }

        filename = f"llm_response_{timestamp.replace(':', '-').replace('.', '_')}.json"
        log_file = self.session_dir / filename

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)

    def log_validation_error(self, error: Exception, raw_data: Dict[str, Any], context: str = "validation") -> None:
        """
        Log detailed information about validation errors (e.g., IRLite creation failures).

        Args:
            error: The exception that occurred
            raw_data: The raw data that failed validation
            context: Context description for the error
        """
        if not self.enabled:
            return

        timestamp = datetime.now().isoformat()
        log_data = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "step": f"{context}_validation_error",
            "type": "validation_error",
            "error": str(error),
            "error_type": type(error).__name__,
            "raw_data": raw_data,
            "validation_errors": getattr(error, "errors", lambda: [])() if hasattr(error, "errors") else [],
        }

        filename = f"{context}_validation_error_{timestamp.replace(':', '-').replace('.', '_')}.json"
        log_file = self.session_dir / filename

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False, default=str)

    def log_reconciliation_summary(
        self, original_text: str, reconciled_text: str, matched_symbols: List[str], unknown_mentions: List[str], lexicon_hits: List[str]
    ) -> None:
        """
        Log summary of reconciliation process results.

        Args:
            original_text: Original input text
            reconciled_text: Final reconciled text with backticks
            matched_symbols: List of symbols that were successfully matched
            unknown_mentions: List of mentions that couldn't be matched
            lexicon_hits: List of lexicon terms that were used
        """
        if not self.enabled:
            return

        timestamp = datetime.now().isoformat()
        log_data = {
            "timestamp": timestamp,
            "session_id": self.session_id,
            "step": "reconciliation_summary",
            "type": "summary",
            "original_text": original_text,
            "reconciled_text": reconciled_text,
            "matched_symbols": matched_symbols,
            "unknown_mentions": unknown_mentions,
            "lexicon_hits": lexicon_hits,
            "stats": {
                "matched_count": len(matched_symbols),
                "unknown_count": len(unknown_mentions),
                "lexicon_hits_count": len(lexicon_hits),
                "original_length": len(original_text),
                "reconciled_length": len(reconciled_text),
            },
        }

        filename = f"reconciliation_summary_{timestamp.replace(':', '-').replace('.', '_')}.json"
        log_file = self.session_dir / filename

        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2, ensure_ascii=False)


# Global debug logger instance
_debug_logger: Optional[DebugLogger] = None


def get_debug_logger(project_root: str = ".") -> DebugLogger:
    """
    Get or create global debug logger instance.

    Args:
        project_root: Project root directory

    Returns:
        DebugLogger instance
    """
    global _debug_logger
    if _debug_logger is None or _debug_logger.project_root != project_root:
        _debug_logger = DebugLogger(project_root)
    return _debug_logger


def is_debug_enabled() -> bool:
    """
    Check if debug logging is enabled via environment variable.

    Returns:
        True if PD_DEBUG=1 is set (or PD_DEBUG_RECONCILE=1 for backward compatibility)
    """
    return os.getenv("PD_DEBUG", os.getenv("PD_DEBUG_RECONCILE", "0")) == "1"
