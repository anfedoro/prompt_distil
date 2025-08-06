"""
Transcript distillation functionality for the Prompt Distiller.

This module handles the core distillation process of converting raw transcripts
into structured intermediate representation (IR-lite) and then rendering prompts.
Uses OpenAI API for intelligent parsing and structuring of transcript content.
"""

import json
from typing import Dict, Literal, Optional

from .config import config, get_client
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
        lex_mode: Literal["rules", "llm", "hybrid"] = "hybrid",
    ) -> IRLite:
        """
        Convert transcript into structured IR-lite representation.

        Args:
            transcript: Raw transcript text to analyze
            surface_hints: Optional project surface information
            target_language: Target language for final prompts ("en" or "auto")
            asr_language: Language detected by ASR (or "auto")
            lex_mode: Lexicon processing mode ("rules", "llm", "hybrid")

        Returns:
            Structured IRLite object

        Raises:
            DistillationError: If distillation fails
        """
        # Protect code identifiers
        from .speech import protect_code_identifiers

        protected_transcript = protect_code_identifiers(transcript)

        # Reconcile text with known symbols from project cache (pass asr_language and lex_mode)
        reconciled_transcript, reconciled_identifiers, unknown_mentions, lexicon_hits, unresolved_terms = reconcile_text(
            protected_transcript, self.project_root, asr_language, lex_mode
        )

        # Load cache for compact hints
        cache = load_cache(self.project_root) or ensure_cache(self.project_root, save=False)

        # Prepare system prompt with distillation instructions and symbol hints
        compact_hints = self._prepare_compact_hints(cache, surface_hints)
        system_prompt = self._create_system_prompt(compact_hints, target_language)

        # Create user message with reconciled transcript
        user_prompt = self._create_user_prompt(reconciled_transcript)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                response_format={"type": "json_object"},
                temperature=0.1,  # Low temperature for consistent parsing
            )

            # Parse response
            content = response.choices[0].message.content
            if not content:
                raise DistillationError("Empty response from model")

            # Parse JSON and validate with Pydantic
            try:
                data = json.loads(content)
                # Add reconciled identifiers and unknown mentions to the data before creating IRLite
                data["reconciled_identifiers"] = reconciled_identifiers
                data["unknown_identifier_mentions"] = unknown_mentions
                data["lexicon_hits"] = lexicon_hits
                data["unresolved_terms"] = unresolved_terms
                ir = IRLite(**data)
                return ir
            except json.JSONDecodeError as e:
                raise DistillationError(f"Invalid JSON response: {e}")
            except Exception as e:
                raise DistillationError(f"Failed to create IRLite from response: {e}")

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
        lex_mode: Literal["rules", "llm", "hybrid"] = "hybrid",
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
        ir = self.build_ir_lite(transcript, surface_hints, target_language, asr_language, lex_mode)

        # Render all prompts
        prompts = self.render_prompts(ir)

        # Create session passport
        passport = self._create_session_passport(transcript, ir, surface_hints, target_language, asr_language, lex_mode)

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

    def _create_system_prompt(self, compact_hints: Optional[Dict] = None, target_language: Literal["en", "auto"] = "en") -> str:
        """Create system prompt for transcript distillation."""
        base_prompt = (
            """You are an expert at distilling noisy transcripts into structured intent representations for coding agents.

Your task is to analyze a transcript and extract structured information in JSON format that follows this schema:

{
  "goal": "Primary objective or goal (required)",
  "scope_hints": ["Context and scope information"],
  "must": ["Required constraints and behaviors"],
  "must_not": ["Prohibited actions or changes"],
  "known_entities": [
    {
      "path": "file/path/or/module",
      "symbol": "function_or_class_name",
      "confidence": 0.8
    }
  ],
  "unknowns": ["Unclear or ambiguous requirements"],
  "acceptance": ["Acceptance criteria"],
  "assumptions": ["Assumptions made during analysis"]
}

Guidelines:
1. Extract the main goal clearly and concisely
2. Identify specific files, functions, classes mentioned (known_entities)
3. Separate clear requirements (must) from prohibitions (must_not)
4. Flag unclear items as unknowns rather than guessing
5. Convert uncertain facts into explicit assumptions
6. Preserve earlier goals unless explicitly overridden
7. Don't include implementation advice - focus on WHAT, not HOW
8. Use confidence scores (0.0-1.0) for known_entities based on clarity

CRITICAL: Code Identifier Preservation
- Preserve backticked identifiers verbatim; do not translate, rewrite, or pluralize them
- Treat them as code entities (functions/classes/files)
- If a Russian transcript contains code-like tokens (underscores or backticks), keep them as-is in IR-lite and the final English prompts
- Code identifiers like `delete_task`, `login_handler`, `FastAPI` must remain exactly as written

LANGUAGE REQUIREMENTS:
- Produce final prompts in """
            + ("English (default)" if target_language == "en" else "the source language")
            + """
- If source is not English and target is English, translate narrative text only, and preserve backticked identifiers verbatim
- Never translate code identifiers, file paths, or technical symbols

"""
        )

        if compact_hints:
            hint_text = self._format_compact_hints(compact_hints)
            base_prompt += f"\nProject Context:\n{hint_text}\n"

        base_prompt += "\nRespond only with valid JSON matching the schema above."

        return base_prompt

    def _create_user_prompt(self, transcript: str) -> str:
        """Create user prompt containing the transcript to analyze."""
        return f"""Analyze this transcript and extract structured intent information:

TRANSCRIPT:
{transcript}

Extract the information into the JSON schema format as instructed."""

    def _format_compact_hints(self, hints: Dict) -> str:
        """Format compact hints for inclusion in system prompt."""
        lines = []

        if hints.get("top_dirs"):
            lines.append("Top-level directories:")
            for directory in hints["top_dirs"][:5]:  # Top 5
                lines.append(f"  - {directory}")

        if hints.get("sample_files"):
            lines.append("Sample files:")
            for file_path in hints["sample_files"][:10]:  # Top 10
                lines.append(f"  - {file_path}")

        if hints.get("known_symbols"):
            lines.append("Known symbols:")
            symbol_list = ", ".join(hints["known_symbols"][:20])  # Top 20
            lines.append(f"  {symbol_list}")

        return "\n".join(lines)

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
        surface_hints: Optional[Dict] = None,
        target_language: Literal["en", "auto"] = "en",
        asr_language: str = "auto",
        lex_mode: Literal["rules", "llm", "hybrid"] = "hybrid",
    ) -> Dict:
        """
        Create a session passport summarizing processing decisions.

        Args:
            transcript: Original transcript
            ir: Generated IR-lite
            surface_hints: Surface hints used
            target_language: Target language for prompts
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

        # Get lexicon hits from IR
        lexicon_hits = ir.lexicon_hits

        # Get additional data from IR
        unresolved_terms = ir.unresolved_terms
        # Use lex_mode from function parameter since it's not stored in IR

        # Determine effective lexicon language
        from .lexicon import get_effective_language, get_stemmer

        lexicon_lang, lang_detect_meta = get_effective_language(asr_language, transcript, str(self.project_root))
        stemmer_lang = lexicon_lang if get_stemmer(lexicon_lang) else "none"

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
            "target_language": target_language,
            "lexicon_lang": lexicon_lang,
            "lexicon_hits": lexicon_hits,
            "lex_mode": lex_mode,
            "stemmer_lang": stemmer_lang,
            "unresolved_terms": unresolved_terms,
            "lang_detect_meta": lang_detect_meta,
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


# Convenience functions for simple use cases
def build_ir_lite(
    transcript: str,
    surface_hints: Optional[Dict] = None,
    project_root: str = ".",
    target_language: Literal["en", "auto"] = "en",
    asr_language: str = "auto",
    lex_mode: Literal["rules", "llm", "hybrid"] = "hybrid",
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
    return distiller.build_ir_lite(transcript, surface_hints, target_language, asr_language, lex_mode)


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
    lex_mode: Literal["rules", "llm", "hybrid"] = "hybrid",
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
    return distiller.distill_complete(transcript, profile, target_language, asr_language, lex_mode)
