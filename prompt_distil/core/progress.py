"""
Global progress reporting module for the Prompt Distiller.

This module provides a centralized progress reporter that can be used
throughout the application to show meaningful status updates during processing.
"""

from typing import List, Optional

from rich.console import Console
from rich.status import Status


class ProgressReporter:
    """
    Global progress reporter for status updates with step completion tracking.

    Provides a centralized way to report progress steps without passing
    console or status objects through function parameters. Tracks completed
    steps and shows checkmarks for enhanced user experience.
    """

    def __init__(self):
        self._status: Optional[Status] = None
        self._console: Optional[Console] = None
        self._completed_steps: List[str] = []
        self._current_step: Optional[str] = None

    def initialize(self, console: Console, initial_message: str = "Starting...") -> Status:
        """
        Initialize the reporter with a console and create a status object.

        Args:
            console: Rich console instance
            initial_message: Initial status message

        Returns:
            Status object that should be used in a context manager
        """
        self._console = console
        self._status = console.status(f"[dim]{initial_message}[/dim]")
        self._completed_steps = []
        self._current_step = initial_message
        return self._status

    def step(self, message: str) -> None:
        """
        Update the current progress step and mark previous step as completed.

        Args:
            message: Progress step message to display
        """
        if self._status is not None:
            # Mark previous step as completed if exists
            if self._current_step is not None:
                self._completed_steps.append(self._current_step)
                if self._console is not None:
                    self._console.print(f"[green]✓[/green] [dim]{self._current_step}[/dim]")

            # Update to new step
            self._current_step = message
            self._status.update(f"[dim]{message}[/dim]")

    def complete_step(self, message: Optional[str] = None) -> None:
        """
        Mark the current step as completed without starting a new one.

        Args:
            message: Optional custom completion message
        """
        if self._current_step is not None:
            completion_msg = message or self._current_step
            self._completed_steps.append(completion_msg)
            if self._console is not None:
                self._console.print(f"[green]✓[/green] [dim]{completion_msg}[/dim]")
            self._current_step = None

    def step_with_context(self, message: str, context: str = "") -> None:
        """
        Update progress step with additional context information.

        Args:
            message: Main progress step message
            context: Additional context (e.g., "for reconciliation", "for distillation")
        """
        full_message = f"{message} {context}".strip()
        self.step(full_message)

    def sub_step(self, message: str, current: int = 0, total: int = 0) -> None:
        """
        Update progress with sub-step information without marking previous step as completed.

        Args:
            message: Sub-step message to display
            current: Current step number (optional)
            total: Total number of steps (optional)
        """
        if self._status is not None:
            if current > 0 and total > 0:
                progress_msg = f"{message} ({current}/{total})"
            else:
                progress_msg = message
            self._status.update(f"[dim]{progress_msg}[/dim]")

    def sub_step_with_progress(self, base_message: str, sub_message: str, current: int, total: int) -> None:
        """
        Update progress with hierarchical sub-step information.

        Args:
            base_message: Main step message
            sub_message: Sub-step description
            current: Current sub-step number
            total: Total number of sub-steps
        """
        if self._status is not None:
            full_message = f"{base_message} - {sub_message} ({current}/{total})"
            self._status.update(f"[dim]{full_message}[/dim]")

    def complete_sub_step(self, message: str) -> None:
        """
        Mark a sub-step as completed with a persistent checkmark message.

        This prints a completed sub-step without changing the current main step,
        ensuring that detailed reconciliation and processing steps remain visible.

        Args:
            message: Sub-step completion message to display
        """
        if self._console is not None:
            # Print with indented checkmark to show it's a sub-step
            self._console.print(f"  [green]✓[/green] [dim]{message}[/dim]")


# Global reporter instance
reporter = ProgressReporter()
