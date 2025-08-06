"""
Main CLI interface for the Prompt Distiller.

This module provides the Typer-based command-line interface with commands for:
- Text-based distillation
- Audio file processing with Whisper
- Project indexing utilities
"""

import sys
from pathlib import Path
from typing import List, Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

from .core.config import ConfigError, validate_config
from .core.distill import distill_transcript
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
    lex_mode: str = typer.Option("hybrid", "--lex-mode", help="Lexicon mode (rules|llm|hybrid)"),
):
    """
    Distill text transcript into structured prompts.

    Examples:
        prompt-distil distill --text "rewrite delete_task test to cover 404; don't change public API"
        prompt-distil distill --file transcript.txt --profile verbose
        prompt-distil distill --project-root /path/to/app --text "update user model" --profile std
    """
    try:
        # Validate input options
        with console.status("[dim]Validating input…[/dim]"):
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

            # Validate lex_mode
            if lex_mode not in ["rules", "llm", "hybrid"]:
                console.print(f"[bold red]Error:[/bold red] Invalid lex-mode '{lex_mode}'. Must be one of: rules, llm, hybrid")
                sys.exit(1)

            # Validate configuration
            validate_config()

        # Apply code identifier protection
        with console.status("[dim]Protecting identifiers…[/dim]"):
            from .core.speech import protect_code_identifiers

            protected_text = protect_code_identifiers(text)

        # Load or ensure cache for reconciliation
        with console.status("[dim]Building/Using cache…[/dim]"):
            if not load_cache(project_root):
                ensure_cache(project_root, save=False)

        # Reconciliation step
        with console.status("[dim]Reconciling…[/dim]"):
            pass  # Reconciliation happens inside distill_transcript

        # Process transcript (always use English for text input)
        with console.status("[dim]Calling the model…[/dim]"):
            result = distill_transcript(protected_text, profile, project_root, "en", "auto", lex_mode)

        # Render results
        with console.status("[dim]Rendering…[/dim]"):
            pass  # Rendering happens inside distill_transcript

        # Display results
        with console.status("[dim]Printing results…[/dim]"):
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
    lex_mode: str = typer.Option("hybrid", "--lex-mode", help="Lexicon mode (rules|llm|hybrid)"),
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
        # Validate input
        with console.status("[dim]Validating input…[/dim]"):
            if lex_mode not in ["rules", "llm", "hybrid"]:
                console.print(f"[bold red]Error:[/bold red] Invalid lex-mode '{lex_mode}'. Must be one of: rules, llm, hybrid")
                sys.exit(1)

            # Validate configuration
            validate_config()

        # Check audio file
        with console.status("[dim]Checking audio file…[/dim]"):
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
        with console.status("[dim]Transcribing audio…[/dim]"):
            transcript_result = speech_processor.transcribe_audio(path)

        transcript_text = transcript_result.text
        console.print(f"[dim]Transcript ({len(transcript_text.split())} words, detected: {transcript_result.lang_hint}):[/dim]")
        console.print(Panel(transcript_text[:200] + "..." if len(transcript_text) > 200 else transcript_text))

        # Load or ensure cache for reconciliation
        with console.status("[dim]Building/Using cache…[/dim]"):
            if not load_cache(project_root):
                ensure_cache(project_root, save=False)

        # Determine target language
        target_language = "en" if translate or final_lang == "en" else "auto"

        # Protecting identifiers step
        with console.status("[dim]Protecting identifiers…[/dim]"):
            pass  # Protection happens inside distill_transcript

        # Reconciliation step
        with console.status("[dim]Reconciling…[/dim]"):
            pass  # Reconciliation happens inside distill_transcript

        # Distill transcript
        with console.status("[dim]Calling the model…[/dim]"):
            result = distill_transcript(transcript_text, profile, project_root, target_language, transcript_result.lang_hint, lex_mode)

        # Update session passport with ASR language info
        result["session_passport"]["asr_language"] = transcript_result.lang_hint

        # Render results
        with console.status("[dim]Rendering…[/dim]"):
            pass  # Rendering happens inside distill_transcript

        # Display results
        with console.status("[dim]Printing results…[/dim]"):
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
            with console.status(f"[dim]Searching for '{search}'…[/dim]"):
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
            with console.status("[dim]Building/Using cache…[/dim]"):
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

    if output_format == "json":
        import json

        # Convert IRLite to dict for JSON serialization
        ir_dict = result["ir"].model_dump()
        output = {"ir": ir_dict, "prompts": result["prompts"], "session_passport": result["session_passport"]}
        console.print(json.dumps(output, indent=2))
        return

    elif output_format == "markdown":
        console.print(result["selected_prompt"])
        return

    # Rich format (default)
    passport = result["session_passport"]

    # Display selected prompt
    console.print(f"\n[bold green]Generated Prompt ({profile} profile):[/bold green]")
    syntax = Syntax(result["selected_prompt"], "markdown", theme="monokai", line_numbers=False)
    console.print(Panel(syntax, border_style="green"))

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

    # Show lexicon information
    if passport.get("lexicon_lang"):
        passport_table.add_row("Lexicon language", passport["lexicon_lang"])

    if passport.get("lexicon_hits"):
        lexicon_terms = ", ".join(passport["lexicon_hits"])
        passport_table.add_row("Lexicon hits", f"`{lexicon_terms}`")

    if passport.get("lex_mode"):
        passport_table.add_row("Lexicon mode", passport["lex_mode"])

    if passport.get("stemmer_lang"):
        passport_table.add_row("Stemmer language", passport["stemmer_lang"])

    if passport.get("unresolved_terms"):
        unresolved = ", ".join(passport["unresolved_terms"])
        passport_table.add_row("Unresolved terms", f"`{unresolved}`")

    # Show project information
    if passport.get("project_root"):
        passport_table.add_row("Project root", passport["project_root"])

    # Show language information
    if passport.get("asr_language"):
        passport_table.add_row("ASR language", passport["asr_language"])
    if passport.get("target_language"):
        passport_table.add_row("Target language", passport["target_language"])

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
