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

    def is_active(self) -> bool:
        """
        Check if the reporter is currently active.

        Returns:
            True if reporter is initialized and active
        """
        return self._status is not None

    def reset(self) -> None:
        """Reset the reporter state."""
        # Complete any remaining step
        if self._current_step is not None and self._console is not None:
            self._console.print(f"[green]✓[/green] [dim]{self._current_step}[/dim]")

        self._status = None
        self._console = None
        self._completed_steps = []
        self._current_step = None

    def get_completed_steps(self) -> List[str]:
        """
        Get list of completed steps.

        Returns:
            List of completed step messages
        """
        return self._completed_steps.copy()


# Global reporter instance
reporter = ProgressReporter()
