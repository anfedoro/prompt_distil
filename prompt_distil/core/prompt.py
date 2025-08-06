"""
Prompt rendering functionality for the Prompt Distiller.

This module provides deterministic templates that render Short, Standard, and Verbose
prompts from the IR-lite structure with sections like Goal, Context, Change request,
Constraints, Acceptance, Assumptions, Out of scope, and Deliverables.
"""

from typing import Dict, List

from .types import IRLite, KnownEntity


class PromptRenderer:
    """
    Renders prompts from IR-lite structure in different verbosity levels.

    Supports three profiles:
    - Short: Essential information only
    - Standard: Balanced detail with all key sections
    - Verbose: Comprehensive with extended context and explanations
    """

    def __init__(self):
        self.profiles = {
            "short": self._render_short,
            "std": self._render_standard,
            "standard": self._render_standard,
            "verbose": self._render_verbose,
        }

    def render(self, ir: IRLite, profile: str = "standard") -> str:
        """
        Render a prompt from IR-lite structure.

        Args:
            ir: The intermediate representation to render
            profile: Rendering profile (short, standard, verbose)

        Returns:
            Rendered prompt as string

        Raises:
            ValueError: If profile is not supported
        """
        if profile not in self.profiles:
            raise ValueError(f"Unsupported profile: {profile}. Available: {list(self.profiles.keys())}")

        return self.profiles[profile](ir)

    def render_all(self, ir: IRLite) -> Dict[str, str]:
        """
        Render prompts for all supported profiles.

        Args:
            ir: The intermediate representation to render

        Returns:
            Dictionary with profile names as keys and rendered prompts as values
        """
        return {profile: self.render(ir, profile) for profile in self.profiles.keys()}

    def _render_short(self, ir: IRLite) -> str:
        """Render short profile with essential information only."""
        sections = []

        # Goal (required)
        sections.append(f"## Goal\n{ir.goal}")

        # Context (if scope hints available)
        if ir.scope_hints:
            context = "\n".join(f"- {hint}" for hint in ir.scope_hints)
            sections.append(f"## Context\n{context}")

        # Change request (combines must/must_not)
        change_items = []
        if ir.must:
            change_items.extend(f"✓ {item}" for item in ir.must)
        if ir.must_not:
            change_items.extend(f"✗ {item}" for item in ir.must_not)

        if change_items:
            changes = "\n".join(change_items)
            sections.append(f"## Change Request\n{changes}")

        # Known entities (if any)
        if ir.known_entities:
            entities = self._format_entities(ir.known_entities, compact=True)
            sections.append(f"## Related (if any)\n{entities}")

        return "\n\n".join(sections)

    def _render_standard(self, ir: IRLite) -> str:
        """Render standard profile with balanced detail."""
        sections = []

        # Goal
        sections.append(f"## Goal\n{ir.goal}")

        # Context
        if ir.scope_hints:
            context = "\n".join(f"- {hint}" for hint in ir.scope_hints)
            sections.append(f"## Context\n{context}")

        # Change request
        change_items = []
        if ir.must:
            change_items.append("**Required:**")
            change_items.extend(f"- {item}" for item in ir.must)
        if ir.must_not:
            if change_items:
                change_items.append("")
            change_items.append("**Prohibited:**")
            change_items.extend(f"- {item}" for item in ir.must_not)

        if change_items:
            changes = "\n".join(change_items)
            sections.append(f"## Change Request\n{changes}")

        # Constraints (unknowns become constraints)
        constraints = []
        if ir.unknowns:
            constraints.append("**Unclear requirements (handle carefully):**")
            constraints.extend(f"- {item}" for item in ir.unknowns)

        if constraints:
            constraint_text = "\n".join(constraints)
            sections.append(f"## Constraints\n{constraint_text}")

        # Acceptance criteria
        if ir.acceptance:
            acceptance = "\n".join(f"- {criterion}" for criterion in ir.acceptance)
            sections.append(f"## Acceptance Criteria\n{acceptance}")

        # Assumptions
        if ir.assumptions:
            assumptions = "\n".join(f"- {assumption}" for assumption in ir.assumptions)
            sections.append(f"## Assumptions\n{assumptions}")

        # Known entities
        if ir.known_entities:
            entities = self._format_entities(ir.known_entities, compact=False)
            sections.append(f"## Related (if any)\n{entities}")

        return "\n\n".join(sections)

    def _render_verbose(self, ir: IRLite) -> str:
        """Render verbose profile with comprehensive detail."""
        sections = []

        # Goal with expanded context
        goal_section = [ir.goal]
        if ir.scope_hints:
            goal_section.append("")
            goal_section.append("**Project Context:**")
            goal_section.extend(f"- {hint}" for hint in ir.scope_hints)

        sections.append("## Goal\n" + "\n".join(goal_section))

        # Detailed change request
        if ir.must or ir.must_not:
            change_parts = []

            if ir.must:
                change_parts.append("### Required Changes")
                change_parts.extend(f"- {item}" for item in ir.must)
                change_parts.append("")

            if ir.must_not:
                change_parts.append("### Prohibited Changes")
                change_parts.extend(f"- {item}" for item in ir.must_not)

            sections.append("## Change Request\n" + "\n".join(change_parts).rstrip())

        # Constraints and considerations
        constraint_parts = []

        if ir.unknowns:
            constraint_parts.append("### Unclear Requirements")
            constraint_parts.append("The following items need clarification or careful handling:")
            constraint_parts.extend(f"- {item}" for item in ir.unknowns)
            constraint_parts.append("")

        constraint_parts.append("### General Constraints")
        constraint_parts.append("- Maintain existing public APIs unless explicitly requested")
        constraint_parts.append("- Follow existing code patterns and conventions")
        constraint_parts.append("- Ensure backward compatibility where possible")

        sections.append("## Constraints\n" + "\n".join(constraint_parts))

        # Detailed acceptance criteria
        if ir.acceptance:
            acceptance_parts = ["### Functional Requirements"]
            acceptance_parts.extend(f"- {criterion}" for criterion in ir.acceptance)
            acceptance_parts.append("")
            acceptance_parts.append("### Quality Requirements")
            acceptance_parts.append("- Code should be well-tested")
            acceptance_parts.append("- Documentation should be updated if needed")
            acceptance_parts.append("- No regressions in existing functionality")

            sections.append("## Acceptance Criteria\n" + "\n".join(acceptance_parts))

        # Assumptions with rationale
        if ir.assumptions:
            assumption_parts = ["The following assumptions were made during analysis:"]
            assumption_parts.extend(f"- {assumption}" for assumption in ir.assumptions)
            sections.append("## Assumptions\n" + "\n".join(assumption_parts))

        # Out of scope
        out_of_scope = [
            "- Performance optimizations not explicitly requested",
            "- Major architectural changes unless necessary",
            "- Changes to external dependencies",
            "- UI/UX modifications unless specified",
        ]
        sections.append("## Out of Scope\n" + "\n".join(out_of_scope))

        # Deliverables
        deliverables = [
            "- Modified source code files",
            "- Updated or new tests as appropriate",
            "- Documentation updates if needed",
            "- Clear commit messages explaining changes",
        ]
        sections.append("## Deliverables\n" + "\n".join(deliverables))

        # Known entities with detailed information
        if ir.known_entities:
            entities = self._format_entities(ir.known_entities, compact=False, detailed=True)
            sections.append(f"## Related (if any)\n{entities}")

        return "\n\n".join(sections)

    def _format_entities(self, entities: List[KnownEntity], compact: bool = False, detailed: bool = False) -> str:
        """
        Format known entities for display.

        Args:
            entities: List of known entities
            compact: Whether to use compact formatting
            detailed: Whether to include detailed information

        Returns:
            Formatted string representation of entities
        """
        if not entities:
            return ""

        formatted = []
        for entity in entities:
            # Only render if we have:
            # 1. A valid file path (not directory, no placeholders)
            # 2. Confidence >= 0.80
            # 3. Known lineno if available
            if (
                entity.path
                and entity.path not in ["", "unknown", "****", "**", "*"]
                and not entity.path.endswith("/")  # Not a directory
                and "/" in entity.path
                and "." in entity.path.split("/")[-1]  # Has file extension
                and entity.confidence is not None
                and entity.confidence >= 0.80
            ):
                if entity.symbol:
                    line = f"- {entity.path} — `{entity.symbol}` (confidence: {entity.confidence:.2f})"
                else:
                    line = f"- {entity.path} (confidence: {entity.confidence:.2f})"

                formatted.append(line)
            # Skip entities that don't meet criteria - no placeholders in Related section

        return "\n".join(formatted)
