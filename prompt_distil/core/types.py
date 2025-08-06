"""
Type definitions for the Prompt Distiller.

This module defines the intermediate representation (IR-lite) data structures
used to capture and structure information extracted from transcripts.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


class KnownEntity(BaseModel):
    """
    Represents a known entity (file, function, class, etc.) mentioned in the transcript.

    Attributes:
        path: File path or module path
        symbol: Optional symbol name (function, class, variable, etc.)
        confidence: Optional confidence score (0.0 to 1.0)
    """

    path: str = Field(..., description="File path or module path")
    symbol: Optional[str] = Field(default=None, description="Symbol name (function, class, etc.)")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Confidence score")


class IRLite(BaseModel):
    """
    Lightweight intermediate representation of distilled intent from transcript.

    This structure captures the essential information needed to generate
    prompts at different verbosity levels.
    """

    goal: str = Field(..., description="Primary objective or goal")
    scope_hints: List[str] = Field(default_factory=list, description="Scope and context hints")
    must: List[str] = Field(default_factory=list, description="Required constraints and behaviors")
    must_not: List[str] = Field(default_factory=list, description="Prohibited actions or changes")
    known_entities: List[KnownEntity] = Field(default_factory=list, description="Known files, functions, etc.")
    unknowns: List[str] = Field(default_factory=list, description="Unclear or ambiguous requirements")
    acceptance: List[str] = Field(default_factory=list, description="Acceptance criteria")
    assumptions: List[str] = Field(default_factory=list, description="Assumptions made during distillation")
    reconciled_identifiers: List[str] = Field(default_factory=list, description="Identifiers reconciled from symbol cache")
    unknown_identifier_mentions: List[str] = Field(default_factory=list, description="Unknown identifier mentions not in cache")
    lexicon_hits: List[str] = Field(default_factory=list, description="Source language terms replaced by lexicon")
    unresolved_terms: List[str] = Field(default_factory=list, description="Code-like terms that weren't resolved by rules")


class PromptProfile(BaseModel):
    """
    Configuration for prompt rendering profiles.
    """

    name: str = Field(..., description="Profile name (short, standard, verbose)")
    include_sections: List[str] = Field(..., description="Sections to include in this profile")
    max_length: Optional[int] = Field(default=None, description="Maximum character length")


class Transcript(BaseModel):
    """
    Result of automatic speech recognition.
    """

    text: str = Field(..., description="Transcribed text")
    lang_hint: str = Field(default="auto", description="Detected or hinted language code")


class SurfaceHint(BaseModel):
    """
    Surface-level project information to provide context.
    """

    type: str = Field(..., description="Type of hint (file, directory, pattern)")
    path: str = Field(..., description="Path or pattern")
    description: Optional[str] = Field(default=None, description="Brief description")
