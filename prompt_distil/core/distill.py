"""
Transcript distillation functionality for the Prompt Distiller.

This module handles the core distillation process of converting raw transcripts
into structured intermediate representation (IR-lite) and then rendering prompts.
Uses OpenAI API for intelligent parsing and structuring of transcript content.
"""

from typing import Any, Dict, Literal, Optional

from .config import config, get_client
from .llm_handler import LLMHandlerError, get_llm_handler
from .progress import reporter
from .prompt import PromptRenderer
from .reconcile import reconcile_text
from .surface import ProjectSurface, ensure_cache, load_cache
from .types import IRLite


class DistillationError(Exception):
    """Raised when distillation process fails."""

    pass


class TranscriptDistiller:
    """
    Handles the distillation of transcripts into structured prompts.

    This class orchestrates the process of:
    1. Analyzing transcripts with surface-level project context
    2. Generating structured IR-lite representation
    3. Rendering prompts in different verbosity levels
    """

    def __init__(self, project_root: str = "."):
        """
        Initialize the distiller with project context.

        Args:
            project_root: Root directory for project context analysis
        """
        self.client = get_client()
        self.model = config.distil_model
        self.surface = ProjectSurface(project_root)
        self.renderer = PromptRenderer()
        self.project_root = project_root

    def build_ir_lite(
        self,
        transcript: str,
        surface_hints: Optional[Dict] = None,
        target_language: Literal["en", "auto"] = "en",
        asr_language: str = "auto",
    ) -> IRLite:
        """
        Convert transcript into structured IR-lite representation.

        Args:
            transcript: Raw transcript text to analyze
            surface_hints: Optional project surface information
            target_language: Target language for final prompts ("en" or "auto")
            asr_language: Language detected by ASR (or "auto")

        Returns:
            Structured IRLite object

        Raises:
            DistillationError: If distillation fails
        """
        # Step 1: Protect code identifiers
        reporter.step("Protecting code identifiers…")
        from .speech import protect_code_identifiers

        protected_transcript = protect_code_identifiers(transcript)

        # Step 2: Reconcile text with known symbols - includes LLM calls and n-gram search
        reporter.step("Invoking reconciliation model and performing symbol matching…")
        reconciled_transcript, reconciled_identifiers, unknown_mentions, unresolved_terms = reconcile_text(
            protected_transcript, self.project_root, asr_language
        )

        # Step 3: Load symbol cache for context hints
        reporter.step("Loading project symbol cache…")
        cache = load_cache(self.project_root) or ensure_cache(self.project_root, save=False)

        # Step 4: Prepare system and user prompts for distillation
        # Step 4: Prepare project context hints for LLM handler
        reporter.step("Preparing distillation prompts…")
        compact_hints = self._prepare_compact_hints(cache, surface_hints)

        # Step 5: Clean reconciled transcript before sending to distillation model
        reporter.step("Cleaning backtick symbols for distillation model…")
        cleaned_transcript = self._clean_backticks_for_distillation(reconciled_transcript)
        reporter.complete_sub_step("Cleaned backtick-enclosed words from prompt text")

        try:
            # Use centralized LLM handler for distillation request
            llm_handler = get_llm_handler(self.project_root)
            data = llm_handler.make_distillation_request(cleaned_transcript, compact_hints, target_language)

            # Add reconciled identifiers and unknown mentions to the data before creating IRLite
            data["reconciled_identifiers"] = reconciled_identifiers
            data["unknown_identifier_mentions"] = unknown_mentions
            data["lexicon_hits"] = []
            data["unresolved_terms"] = unresolved_terms

            # Create IRLite from response data
            try:
                ir = IRLite(**data)
                reporter.complete_sub_step("Parsed distillation model response into structured format")
                return ir
            except Exception as e:
                # Log detailed information about the validation failure
                import logging

                from .debug_log import get_debug_logger

                logger = logging.getLogger(__name__)
                logger.error("=" * 60)
                logger.error("IRLite VALIDATION FAILURE")
                logger.error("=" * 60)
                logger.error(f"Exception: {e}")
                logger.error(f"Exception type: {type(e).__name__}")

                # Log specific problematic fields if it's a Pydantic validation error
                try:
                    from pydantic import ValidationError  # type: ignore
                except Exception:
                    ValidationError = tuple()  # type: ignore[assignment]
                if isinstance(e, ValidationError):
                    logger.error("Pydantic validation errors:")
                    errors_method = getattr(e, "errors", None)
                    if callable(errors_method):
                        errors_list: list[Dict[str, Any]] = list(errors_method() or [])  # type: ignore[arg-type]
                        for error in errors_list:
                            field_path = " -> ".join(str(loc) for loc in error.get("loc", []))
                            error_type = error.get("type", "unknown")
                            error_msg = error.get("msg", "no message")
                            input_value = error.get("input", "not provided")
                            logger.error(f"  Field: {field_path}")
                            logger.error(f"  Error: {error_type} - {error_msg}")
                            logger.error(f"  Input value: {input_value} (type: {type(input_value).__name__})")
                            logger.error("  ---")
                    else:
                        logger.error("  (no detailed validation errors available)")

                logger.error("=" * 60)

                # Use debug logger for detailed file logging
                debug_logger = get_debug_logger(self.project_root)
                debug_logger.log_validation_error(e, data, "irlite")

                raise DistillationError(f"Failed to create IRLite from response: {e}")

        except LLMHandlerError as e:
            raise DistillationError(f"LLM Handler failed: {e}")
        except Exception as e:
            if isinstance(e, DistillationError):
                raise
            raise DistillationError(f"API call failed: {e}")

    def render_prompts(self, ir: IRLite) -> Dict[str, str]:
        """
        Render prompts from IR-lite in all supported profiles.

        Args:
            ir: The intermediate representation to render

        Returns:
            Dictionary with profile names as keys and rendered prompts as values
        """
        return self.renderer.render_all(ir)

    def distill_complete(
        self,
        transcript: str,
        profile: str = "standard",
        target_language: Literal["en", "auto"] = "en",
        asr_language: str = "auto",
    ) -> Dict:
        """
        Complete distillation pipeline from transcript to final prompt.

        Args:
            transcript: Raw transcript to process
            profile: Rendering profile (short, standard, verbose)
            target_language: Target language for final prompts

        Returns:
            Dictionary containing:
            - ir: The intermediate representation
            - prompts: All rendered prompts
            - selected_prompt: The prompt for requested profile
            - session_passport: Summary of processing decisions
        """
        # Gather surface hints if project context is available
        surface_hints = self._gather_surface_hints()

        # Build IR-lite
        ir = self.build_ir_lite(transcript, surface_hints, target_language, asr_language)

        # Render all prompts
        reporter.step("Rendering final prompts…")
        prompts = self.render_prompts(ir)

        # Create session passport
        reporter.step("Creating session summary…")
        passport = self._create_session_passport(transcript, ir, surface_hints, asr_language)

        return {"ir": ir, "prompts": prompts, "selected_prompt": prompts.get(profile, prompts["standard"]), "session_passport": passport}

    def _prepare_compact_hints(self, cache: Dict, surface_hints: Optional[Dict] = None) -> Dict:
        """Prepare compact hints for LLM from cache and surface hints."""
        hints = {}

        # Top-level directories
        if cache.get("files"):
            dirs = set()
            for file_info in cache["files"][:50]:  # Limit files considered
                path_parts = file_info["path"].split("/")
                if len(path_parts) > 1:
                    dirs.add(path_parts[0])
            hints["top_dirs"] = sorted(list(dirs))[:10]  # Top 10 dirs

        # Sample filenames
        if cache.get("files"):
            sample_files = [f["path"] for f in cache["files"][:20]]
            hints["sample_files"] = sample_files

        # Known symbols (names only, up to K)
        if cache.get("symbols"):
            symbol_names = [s["name"] for s in cache["symbols"][:50]]  # Top 50 symbols
            hints["known_symbols"] = symbol_names

        # Merge with surface hints if provided
        if surface_hints:
            hints.update(surface_hints)

        return hints

    def _gather_surface_hints(self) -> Dict:
        """Gather surface-level project information for context."""
        try:
            # Get project structure overview
            structure = self.surface.get_project_structure(max_depth=2)

            # Get common file patterns
            common_files = []
            for file_path in structure.get("files", [])[:20]:  # Limit to first 20
                if any(file_path.endswith(ext) for ext in [".py", ".js", ".ts", ".java", ".cpp", ".h"]):
                    common_files.append(file_path)

            return {
                "project_root": structure.get("root", ""),
                "directories": structure.get("directories", [])[:10],  # Top 10 dirs
                "common_files": common_files,
                "file_types": structure.get("file_types", {}),
            }

        except Exception:
            # If surface analysis fails, return empty hints
            return {}

    def _format_surface_hints(self, hints: Dict) -> str:
        """Format surface hints for inclusion in system prompt."""
        lines = []

        if hints.get("project_root"):
            lines.append(f"Project root: {hints['project_root']}")

        if hints.get("directories"):
            lines.append("Key directories:")
            for directory in hints["directories"][:5]:  # Top 5
                lines.append(f"  - {directory}")

        if hints.get("common_files"):
            lines.append("Common files:")
            for file_path in hints["common_files"][:10]:  # Top 10
                lines.append(f"  - {file_path}")

        if hints.get("file_types"):
            types_str = ", ".join(f"{ext}({count})" for ext, count in list(hints["file_types"].items())[:5])
            lines.append(f"File types: {types_str}")

        return "\n".join(lines)

    def _create_session_passport(
        self,
        transcript: str,
        ir: IRLite,
        surface_hints: Optional[Dict],
        asr_language: str,
    ) -> Dict[str, Any]:
        """
        Create a session passport summarizing processing decisions.

        Args:
            transcript: Original transcript
            ir: Generated IR-lite
            surface_hints: Surface hints used
            asr_language: Detected or hinted language from ASR

        Returns:
            Dictionary with session information
        """
        transcript_stats = {"length": len(transcript), "words": len(transcript.split()), "lines": transcript.count("\n") + 1}

        processing_stats = {
            "known_entities_found": len(ir.known_entities),
            "unknowns_identified": len(ir.unknowns),
            "assumptions_made": len(ir.assumptions),
            "requirements_extracted": len(ir.must),
            "prohibitions_identified": len(ir.must_not),
        }

        context_info = {"project_context_used": surface_hints is not None, "surface_hints_count": len(surface_hints) if surface_hints else 0}

        # Identify what might have been dropped or simplified
        dropped_info = []
        if transcript_stats["words"] > 200 and processing_stats["requirements_extracted"] < 3:
            dropped_info.append("Large transcript may have had details consolidated")
        if ir.unknowns:
            dropped_info.append(f"{len(ir.unknowns)} unclear items flagged for review")
        if not ir.acceptance:
            dropped_info.append("No explicit acceptance criteria identified")

        # Extract preserved identifiers from transcript and IR
        preserved_identifiers = self._extract_preserved_identifiers(transcript, ir)

        # Get reconciled identifiers from IR
        reconciled_identifiers = ir.reconciled_identifiers

        # Get unknown mentions from IR
        unknown_mentions = ir.unknown_identifier_mentions

        # Get lexicon hits from IR (now empty as lexicon processing is removed)
        lexicon_hits = ir.lexicon_hits

        # Get additional data from IR
        unresolved_terms = ir.unresolved_terms

        return {
            "transcript_stats": transcript_stats,
            "processing_stats": processing_stats,
            "context_info": context_info,
            "dropped_or_simplified": dropped_info,
            "model_used": self.model,
            "assumptions_made": ir.assumptions,
            "preserved_identifiers": preserved_identifiers,
            "reconciled_identifiers": reconciled_identifiers,
            "unknown_identifier_mentions": unknown_mentions,
            "project_root": str(self.project_root),
            "asr_language": asr_language,
        }

    def _extract_preserved_identifiers(self, transcript: str, ir: IRLite) -> list[str]:
        """
        Extract list of preserved code identifiers from transcript and IR.

        Args:
            transcript: Original transcript text
            ir: Generated IR-lite object

        Returns:
            List of preserved code identifiers
        """
        import re

        identifiers = set()

        # Extract backticked identifiers from transcript
        backtick_pattern = r"`([^`]+)`"
        identifiers.update(re.findall(backtick_pattern, transcript))

        # Extract identifiers from IR text fields
        ir_text = " ".join(
            [
                ir.goal,
                " ".join(ir.scope_hints),
                " ".join(ir.must),
                " ".join(ir.must_not),
                " ".join(ir.unknowns),
                " ".join(ir.acceptance),
                " ".join(ir.assumptions),
            ]
        )
        identifiers.update(re.findall(backtick_pattern, ir_text))

        # Extract symbols from known entities
        for entity in ir.known_entities:
            if entity.symbol:
                identifiers.add(entity.symbol)

        return sorted(list(identifiers))

    def _clean_backticks_for_distillation(self, text: str) -> str:
        """
        Clean backtick-enclosed words from text before sending to distillation model.

        Removes backticks while preserving the identifiers for proper processing
        by the distillation model.

        Args:
            text: Text with potential backtick-enclosed identifiers

        Returns:
            Cleaned text with backticks removed but identifiers preserved
        """
        import re

        # Remove backticks but preserve the enclosed content
        # This ensures identifiers are clean for the distillation model
        cleaned_text = re.sub(r"`([^`]+)`", r"\1", text)

        return cleaned_text


# Convenience functions for simple use cases
def build_ir_lite(
    transcript: str,
    surface_hints: Optional[Dict] = None,
    project_root: str = ".",
    target_language: Literal["en", "auto"] = "en",
    asr_language: str = "auto",
) -> IRLite:
    """
    Convenience function to build IR-lite from transcript.

    Args:
        transcript: Raw transcript to analyze
        surface_hints: Optional surface hints
        project_root: Project root for context
        target_language: Target language for prompts

    Returns:
        IRLite object
    """
    distiller = TranscriptDistiller(project_root)
    return distiller.build_ir_lite(transcript, surface_hints, target_language, asr_language)


def render_prompts(ir: IRLite) -> Dict[str, str]:
    """
    Convenience function to render prompts from IR-lite.

    Args:
        ir: IR-lite object to render

    Returns:
        Dictionary of rendered prompts
    """
    renderer = PromptRenderer()
    return renderer.render_all(ir)


def distill_transcript(
    transcript: str,
    profile: str = "standard",
    project_root: str = ".",
    target_language: Literal["en", "auto"] = "en",
    asr_language: str = "auto",
) -> Dict:
    """
    Complete convenience function for transcript distillation.

    Args:
        transcript: Raw transcript to process
        profile: Rendering profile
        project_root: Project root
        target_language: Target language for prompts

    Returns:
        Complete distillation results
    """
    distiller = TranscriptDistiller(project_root)
    return distiller.distill_complete(transcript, profile, target_language, asr_language)
