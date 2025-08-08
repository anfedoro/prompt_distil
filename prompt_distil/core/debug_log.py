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
from typing import List, Optional


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
            enabled: Override debug enable flag, uses PD_DEBUG_RECONCILE env var if None
        """
        self.project_root = project_root
        self.enabled = enabled if enabled is not None else os.getenv("PD_DEBUG_RECONCILE") == "1"
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
        True if PD_DEBUG_RECONCILE=1 is set
    """
    return os.getenv("PD_DEBUG_RECONCILE") == "1"
