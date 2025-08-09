"""
Main CLI interface for the Prompt Distiller.

This module provides the Typer-based command-line interface with commands for:
- Text-based distillation
- Audio file processing with Whisper
- Project indexing utilities
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

import pyperclip
import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .core.config import ConfigError, validate_config
from .core.distill import distill_transcript
from .core.progress import reporter
from .core.speech import SpeechError, SpeechProcessor
from .core.surface import ProjectSurface, SurfaceError, ensure_cache, load_cache

app = typer.Typer(
    name="prompt-distil",
    help="Intent Distiller CLI - Turn noisy transcripts into clean prompts for coding agents",
    no_args_is_help=True,
)

console = Console()


@app.command()
def distill(
    text: Optional[str] = typer.Option(None, "--text", "-t", help="Text transcript to distill"),
    file: Optional[str] = typer.Option(None, "--file", help="Path to file containing transcript text"),
    profile: str = typer.Option("standard", "--profile", "-p", help="Rendering profile (short|std|standard|verbose)"),
    project_root: str = typer.Option(".", "--project-root", help="Project root directory for symbol cache and context"),
    output_format: str = typer.Option("rich", "--format", "-f", help="Output format (rich, markdown, json)"),
    debug: bool = typer.Option(False, "--debug", help="Enable detailed debug logging for reconcile_text hybrid mode"),
):
    """
    Distill text transcript into structured prompts.

    Examples:
        prompt-distil distill --text "rewrite delete_task test to cover 404; don't change public API"
        prompt-distil distill --file transcript.txt --profile verbose
        prompt-distil distill --project-root /path/to/app --text "update user model" --profile std
    """
    try:
        # Initialize progress reporter
        with reporter.initialize(console, "Validating input…"):
            # Validate input options
            if text and file:
                console.print("[bold red]Error:[/bold red] Cannot specify both --text and --file options")
                sys.exit(1)

            if not text and not file:
                console.print("[bold red]Error:[/bold red] Must specify either --text or --file option")
                sys.exit(1)

            # Read transcript from file if specified
            if file:
                file_path = Path(file)
                if not file_path.exists():
                    console.print(f"[bold red]Error:[/bold red] File not found: {file}")
                    sys.exit(1)

                try:
                    text = file_path.read_text(encoding="utf-8")
                except Exception as e:
                    console.print(f"[bold red]Error:[/bold red] Failed to read file '{file}': {e}")
                    sys.exit(1)

            # Ensure text is not None at this point
            assert text is not None, "Text should not be None after validation"

            # Validate configuration
            validate_config()

            # Set debug environment variable - CLI flag always overrides .env
            if debug:
                os.environ["PD_DEBUG"] = "1"
            else:
                # Ensure debug is off if flag is explicitly False (overrides .env)
                if "PD_DEBUG" not in os.environ or os.environ.get("PD_DEBUG") == "0":
                    os.environ["PD_DEBUG"] = "0"

            # Load or ensure cache for reconciliation
            reporter.step("Building/Using cache…")
            if not load_cache(project_root):
                ensure_cache(project_root, save=False)

            # Begin distillation process with detailed status reporting
            reporter.step("Starting transcript distillation process…")
            result = distill_transcript(text, profile, project_root, "en", "auto")

            # Display results
            reporter.step("Printing results…")
            reporter.complete_step()
            _display_distillation_result(result, profile, output_format)

    except ConfigError as e:
        console.print(f"[bold red]Configuration Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


@app.command("from-audio")
def from_audio(
    path: str = typer.Argument(..., help="Path to audio file"),
    translate: bool = typer.Option(False, "--translate", help="Generate final prompts in English (regardless of source language)"),
    final_lang: str = typer.Option("en", "--final-lang", help="Final prompt language (en, auto)"),
    profile: str = typer.Option("standard", "--profile", "-p", help="Rendering profile (short|std|standard|verbose)"),
    project_root: str = typer.Option(".", "--project-root", help="Project root directory for symbol cache and context"),
    output_format: str = typer.Option("rich", "--format", "-f", help="Output format (rich, markdown, json)"),
    debug: bool = typer.Option(False, "--debug", help="Enable detailed debug logging for reconcile_text hybrid mode"),
):
    """
    Process audio file with Whisper ASR then distill into prompts.
    ASR uses auto language detection. Final language is controlled by distiller.

    Examples:
        prompt-distil from-audio recording.wav --translate
        prompt-distil from-audio recording.wav --final-lang auto
        prompt-distil from-audio recording.wav --project-root /path/to/app --translate
    """
    try:
        # Initialize progress reporter
        with reporter.initialize(console, "Validating input…"):
            # Validate configuration
            validate_config()

            # Set debug environment variable - CLI flag always overrides .env
            if debug:
                os.environ["PD_DEBUG"] = "1"
            else:
                # Ensure debug is off if flag is explicitly False (overrides .env)
                if "PD_DEBUG" not in os.environ or os.environ.get("PD_DEBUG") == "0":
                    os.environ["PD_DEBUG"] = "0"

            # Check audio file
            reporter.step("Checking audio file…")
            audio_path = Path(path)
            if not audio_path.exists():
                console.print(f"[bold red]Error:[/bold red] Audio file not found: {path}")
                sys.exit(1)

            # Initialize speech processor
            speech_processor = SpeechProcessor()

            # Validate audio format
            if not speech_processor.validate_audio_format(path):
                console.print(f"[bold red]Error:[/bold red] Unsupported audio format: {audio_path.suffix}")
                console.print("Supported formats: .mp3, .mp4, .mpeg, .mpga, .m4a, .wav, .webm")
                sys.exit(1)

            # Show file info
            audio_info = speech_processor.get_audio_info(path)
            console.print(f"[dim]Processing: {audio_info['name']} ({audio_info['size_mb']} MB)[/dim]")

            # Transcribe audio
            transcript_result = speech_processor.transcribe_audio(path)

            transcript_text = transcript_result.text
            console.print(f"[dim]Transcript ({len(transcript_text.split())} words, detected: {transcript_result.lang_hint}):[/dim]")
            console.print(Panel(transcript_text[:200] + "..." if len(transcript_text) > 200 else transcript_text))

            # Load or ensure cache for reconciliation
            reporter.step("Building/Using cache…")
            if not load_cache(project_root):
                ensure_cache(project_root, save=False)

            # Determine target language
            target_language = "en" if translate or final_lang == "en" else "auto"

            # Begin distillation process with detailed status reporting
            reporter.step("Starting transcript distillation process…")
            result = distill_transcript(transcript_text, profile, project_root, target_language, transcript_result.lang_hint)

            # Update session passport with ASR language info
            result["session_passport"]["asr_language"] = transcript_result.lang_hint

            # Display results
            reporter.step("Printing results…")
            reporter.complete_step()
            _display_distillation_result(result, profile, output_format)

    except (ConfigError, SpeechError) as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Unexpected Error:[/bold red] {e}")
        sys.exit(1)


@app.command()
def index(
    patterns: Optional[List[str]] = typer.Option(None, "--glob", "-g", help="Glob patterns for file filtering"),
    project_root: str = typer.Option(".", "--project-root", help="Project root directory for symbol cache"),
    search: Optional[str] = typer.Option(None, "--search", "-s", help="Search query for file content"),
    max_results: int = typer.Option(10, "--max", "-m", help="Maximum results to show"),
    force: bool = typer.Option(False, "--force/--no-force", help="Force rebuild cache"),
    save: bool = typer.Option(True, "--save/--no-save", help="Save cache after building"),
):
    """
    Build symbol cache and perform project indexing utilities.

    Examples:
        prompt-distil index --save
        prompt-distil index --project-root /path/to/app --save
        prompt-distil index --glob "**/*.py" --glob "**/*.js"
        prompt-distil index --search "def test_"
        prompt-distil index --force --save
    """
    try:
        if search:
            # Initialize surface for search
            surface = ProjectSurface(project_root)
            # Perform content search
            with reporter.initialize(console, f"Searching for '{search}'…"):
                results = surface.search_project(search, max_results)

            if not results:
                console.print(f"[yellow]No results found for '{search}'[/yellow]")
                return

            console.print(f"[bold green]Found {len(results)} results:[/bold green]")
            for i, result in enumerate(results, 1):
                console.print(f"\n[bold]{i}. {result['path']}:{result['line_number']}[/bold]")
                console.print(Panel(result["snippet"], border_style="dim"))

        else:
            # Build/update symbol cache
            with reporter.initialize(console, "Building/Using cache…"):
                cache = ensure_cache(project_root, globs=patterns, force=force, save=save)

            # Display cache statistics
            console.print("[bold green]Symbol cache built successfully[/bold green]")

            stats_table = Table(title="Cache Statistics")
            stats_table.add_column("Metric", style="cyan")
            stats_table.add_column("Value", style="white")

            stats_table.add_row("Files processed", str(len(cache.get("files", []))))
            stats_table.add_row("Symbols found", str(len(cache.get("symbols", []))))
            stats_table.add_row("Generated at", cache.get("generated_at", "unknown"))
            stats_table.add_row("Cache saved", "Yes" if save else "No")

            console.print(stats_table)

            # Show sample symbols
            symbols = cache.get("symbols", [])
            if symbols:
                console.print("\n[bold]Sample symbols found:[/bold]")
                for symbol in symbols[:10]:
                    console.print(f"  • {symbol['name']} ({symbol['kind']}) in {symbol['path']}:{symbol['lineno']}")

                if len(symbols) > 10:
                    console.print(f"  ... and {len(symbols) - 10} more symbols")

    except SurfaceError as e:
        console.print(f"[bold red]Surface Error:[/bold red] {e}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {e}")
        sys.exit(1)


def _display_distillation_result(result: dict, profile: str, output_format: str):
    """Display distillation results in the specified format."""

    # Get the prompt text for clipboard copying
    selected_prompt = result["selected_prompt"]

    if output_format == "json":
        import json

        # Convert IRLite to dict for JSON serialization
        ir_dict = result["ir"].model_dump()
        output = {"ir": ir_dict, "prompts": result["prompts"], "session_passport": result["session_passport"]}
        json_output = json.dumps(output, indent=2)
        console.print(json_output)

        # Copy the prompt part in JSON format to clipboard
        prompt_json = json.dumps({"prompt": selected_prompt}, indent=2)
        try:
            pyperclip.copy(prompt_json)
        except Exception:
            # Silently handle clipboard errors to not break functionality
            pass
        return

    elif output_format == "markdown":
        console.print(selected_prompt)

        # Copy the prompt in markdown format to clipboard
        try:
            pyperclip.copy(selected_prompt)
        except Exception:
            # Silently handle clipboard errors to not break functionality
            pass
        return

    # Rich format (default)
    passport = result["session_passport"]

    # Display selected prompt
    console.print(f"\n[bold green]Generated Prompt ({profile} profile):[/bold green]")
    syntax = Syntax(result["selected_prompt"], "markdown", theme="monokai", line_numbers=False)
    console.print(Panel(syntax, border_style="green"))

    # Copy the prompt in markdown format to clipboard for rich format
    try:
        pyperclip.copy(selected_prompt)
    except Exception:
        # Silently handle clipboard errors to not break functionality
        pass

    # Display session passport
    console.print("\n[bold blue]Session Passport:[/bold blue]")

    passport_table = Table(show_header=False, box=None)
    passport_table.add_column("Key", style="cyan")
    passport_table.add_column("Value", style="white")

    stats = passport["processing_stats"]
    passport_table.add_row("Model used", passport["model_used"])
    passport_table.add_row("Known entities", str(stats["known_entities_found"]))
    passport_table.add_row("Requirements", str(stats["requirements_extracted"]))
    passport_table.add_row("Unknowns flagged", str(stats["unknowns_identified"]))
    passport_table.add_row("Assumptions made", str(stats["assumptions_made"]))

    # Show preserved identifiers if any
    if passport.get("preserved_identifiers"):
        preserved = ", ".join(passport["preserved_identifiers"])
        passport_table.add_row("Preserved identifiers", f"`{preserved}`")

    # Show reconciled identifiers if any
    if passport.get("reconciled_identifiers"):
        reconciled = ", ".join(passport["reconciled_identifiers"])
        passport_table.add_row("Reconciled identifiers", f"`{reconciled}`")

    # Show unknown identifier mentions if any
    if passport.get("unknown_identifier_mentions"):
        unknown = ", ".join(passport["unknown_identifier_mentions"])
        passport_table.add_row("Unknown mentions", f"`{unknown}`")

    # Show unknown mentions if any
    if passport.get("unknown_identifier_mentions"):
        unknown = ", ".join(passport["unknown_identifier_mentions"])
        passport_table.add_row("Unknown mentions", f"`{unknown}`")

    # Show project information
    if passport.get("project_root"):
        passport_table.add_row("Project root", passport["project_root"])

    # Show ASR language if relevant
    if passport.get("asr_language") and passport["asr_language"] != "auto":
        passport_table.add_row("ASR language", passport["asr_language"])

    console.print(passport_table)

    # Show dropped/simplified info
    if passport["dropped_or_simplified"]:
        console.print("\n[yellow]Processing Notes:[/yellow]")
        for note in passport["dropped_or_simplified"]:
            console.print(f"  • {note}")

    # Show other profiles available
    other_profiles = [p for p in result["prompts"].keys() if p != profile]
    if other_profiles:
        console.print(f"\n[dim]Other profiles available: {', '.join(other_profiles)}[/dim]")


if __name__ == "__main__":
    app()
