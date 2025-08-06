# Prompt Distiller

Intent Distiller CLI - Turn noisy transcripts into clean prompts for coding agents.

## Overview

Prompt Distiller is a minimal MVP tool that converts noisy voice transcripts into structured, clear English prompts suitable for coding agents. It uses OpenAI's GPT-4 for intelligent parsing and Whisper for automatic speech recognition (ASR).

The tool performs surface-level project understanding (no AST parsing) and generates prompts in three verbosity levels: Short, Standard, and Verbose.

## Features

- **Speech-to-Text**: Convert audio files to text using OpenAI Whisper
- **Intelligent Distillation**: Extract structured intent from noisy transcripts
- **Multiple Output Formats**: Short, Standard, and Verbose prompts
- **Project Context**: Surface-level project indexing for better context
- **Rich CLI Interface**: Beautiful console output with syntax highlighting
- **Multiple Output Formats**: Rich console, Markdown, or JSON output
- **Code Identifier Preservation**: Automatic protection of code identifiers during processing
- **Multilingual Support**: Auto language detection with intelligent translation handling
- **Symbol Cache**: Persistent symbol inventory for fast reconciliation and matching
- **Smart Reconciliation**: Fuzzy matching with stemming and LLM fallback for robust symbol mapping
- **Stemming Support**: Handles inflected forms using Snowball stemmers for ru/es/en
- **LLM Fallback**: Optional AI-assisted mapping when rule-based matching fails

## Installation

This project uses `uv` for dependency management. Make sure you have `uv` installed.

```bash
# Clone the repository
git clone <repository-url>
cd prompt_distil

# Install dependencies
uv add openai pydantic>=2 typer>=0.12 python-dotenv rich

# Add development dependencies (optional)
uv add --dev pytest pytest-asyncio

# Sync dependencies
uv sync
```

## Configuration

Set your OpenAI API key as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

Or create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-api-key-here

# Optional: Override default models
DISTIL_MODEL=gpt-4o
ASR_MODEL=whisper-1
```

## Usage

### Symbol Cache Management

Build and manage the symbol cache for improved reconciliation:

```bash
# Build and save symbol cache for current project
uv run prompt-distil index --project-root . --save

# Build cache for external project
uv run prompt-distil index --project-root /path/to/app --save

# Force rebuild cache
uv run prompt-distil index --project-root . --force --save

# Build cache with custom patterns
uv run prompt-distil index --project-root . --glob "**/*.py" --glob "**/*.js" --save

# Search project content
uv run prompt-distil index --project-root . --search "def test_" --max 10
```

### Text Distillation

Convert text transcripts into structured prompts:

```bash
# Basic usage with text input
uv run prompt-distil distill --text "rewrite delete_task test to cover 404; don't change public API"

# Target specific project for symbol reconciliation
uv run prompt-distil distill --project-root /path/to/app --text "update user_model and login_handler"

# Read transcript from file
uv run prompt-distil distill --file transcript.txt --project-root .

# Choose lexicon mode for symbol matching
uv run prompt-distil distill --text "переписать тест удаления задачи" --lex-mode rules
uv run prompt-distil distill --text "rewrite delete handler" --lex-mode hybrid  # default
uv run prompt-distil distill --text "fix login controller" --lex-mode llm

# Choose output profile (short, std/standard, verbose)
uv run prompt-distil distill --text "add logging to auth flow" --profile short
uv run prompt-distil distill --text "add logging to auth flow" --profile std
uv run prompt-distil distill --text "add logging to auth flow" --profile verbose

# JSON output for programmatic use
uv run prompt-distil distill --text "fix the bug" --format json

# Include project context with explicit root
uv run prompt-distil distill --project-root /path/to/project --text "update user model"
```

### Audio Processing

Process audio files with Whisper ASR then distill:

```bash
# Transcribe and distill audio
uv run prompt-distil from-audio recording.wav

# Generate final prompts in English with project context
uv run prompt-distil from-audio recording.wav --project-root /path/to/app --translate

# Keep final prompts in source language
uv run prompt-distil from-audio recording.wav --final-lang auto

# Use different lexicon modes for audio processing
uv run prompt-distil from-audio russian_audio.wav --lex-mode hybrid --translate

# Different output profile with project context
uv run prompt-distil from-audio recording.wav --project-root . --profile verbose
```

Supported audio formats: `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.wav`, `.webm`

### Project Indexing

Quick project exploration utilities:

```bash
# List Python files
uv run prompt-distil index --glob "**/*.py"

# Search for content
uv run prompt-distil index
 --search "def test_" --max 5

# Explore project structure
uv run prompt-distil index --project /path/to/project
```

## Output Profiles

### Short Profile
- Essential information only
- Goal and basic change request
- Compact formatting

### Standard Profile (Default) - alias: `std`
- Balanced detail
- All key sections: Goal, Context, Change Request, Constraints, Acceptance, Assumptions
- Related files/symbols

### Verbose Profile
- Comprehensive detail
- Extended context and explanations
- Out of scope and deliverables sections
- Quality requirements

## Symbol Cache and Reconciliation

The tool maintains a persistent symbol inventory that enables intelligent reconciliation:

### Cache Features
- **AST-based extraction**: Analyzes Python files to extract functions and classes
- **Persistent storage**: Saves cache to `.prompt_distil/project_cache.json`
- **Bounded scanning**: Limits file count and size to prevent performance issues
- **Inverted index**: Fast lookup of symbols and their locations

### Smart Reconciliation
- **Safe matching**: Only backticks symbols that exist in the project cache
- **Stemming support**: Handles inflected forms (задачи→задача, running→run) using Snowball stemmers
- **Fuzzy matching**: Finds symbols even with typos or variations
- **Alias generation**: Supports CamelCase ↔ snake_case, spaced variations with stemmed forms
- **Context rules**: Special handling for test patterns and common phrases
- **Multilingual lexicons**: Language-aware anchor term mapping for improved matching
- **LLM fallback**: AI-assisted mapping when rule-based matching fails (hybrid/llm modes)
- **Unknown tracking**: Reports mentions that don't match cache symbols

### Symbol Matching Examples
- `delete_task` matches "delete task", "DeleteTask", "deleteTask" (if in cache)
- `LoginHandler` generates "login_handler", "login handler", "LoginHandlerHandler"
- Context: "test for delete task" → `delete_task` with higher confidence
- Russian: "обработчик логина" → `login_handler`
- Spanish: "controlador usuario" → `user_controller`
- **Safe policy**: "unknown_function" → no backticks if not in cache

### Language-aware Reconciliation (Lexicons)
- **Built-in anchor lexicons**: `prompt_distil/data/lexicons/{lang}.json` (ru, es, en stemmers)
- **Per-project overrides**: `{project_root}/.prompt_distil/lexicon/{lang}.json`
- **Auto language detection**: Uses ASR-detected language or fallback heuristics
- **Stemming-aware**: Handles inflected forms (удаления→удалить, задачи→задача)
- **Anchor terms only**: Maps domain terms (e.g., обработчик→handler, удалить→delete)
- **Three modes**:
  - `rules`: Fast deterministic matching with stemming
  - `llm`: AI-assisted mapping for complex cases
  - `hybrid`: Rules first, LLM fallback for unresolved terms (default)
- **Deterministic and safe**: Phrases map to backticked symbols only if they exist in cache
- **Session tracking**: Reports lexicon language, hits, reconciled identifiers, and processing mode

## ASR and Language Handling

The tool uses a sophisticated approach to handle multiple languages while preserving code:

### Automatic Speech Recognition (ASR)
- **Auto Language Detection**: Whisper automatically detects the source language
- **No ASR Translation**: Audio is always transcribed in the original language
- **Supported Formats**: `.mp3`, `.mp4`, `.mpeg`, `.mpga`, `.m4a`, `.wav`, `.webm`

### Final Language Processing
- **Distiller Translation**: Final English prompts are generated by the LLM distiller
- **Code Preservation**: Code identifiers remain intact during translation
- **Language Options**:
  - `--translate`: Generate final prompts in English
  - `--final-lang auto`: Keep prompts in source language
  - `--final-lang en`: Force English output (default)

### Code Identifier Preservation
The tool automatically protects code identifiers by wrapping them in backticks:
- **Underscored identifiers**: `delete_task`, `user_model`, `login_handler`
- **Known frameworks**: `FastAPI`, `pydantic`, `pytest`, `EmailStr`
- **Preserved during translation**: Code stays intact regardless of source language
- **Session tracking**: Shows preserved identifiers in session passport

## Example Output

```bash
uv run prompt-distil distill --project-root /path/to/app --text "rewrite delete_task test to cover 404; don't change public API" --profile std --format markdown
```

**Generated Prompt (standard profile):**
```markdown
## Goal
Rewrite the `delete_task` test to cover a 404 scenario

## Context
- Ensure the public API remains unchanged

## Change Request
**Required:**
- Rewrite the `delete_task` test to include a 404 scenario

**Prohibited:**
- Change the public API

## Constraints
**Unclear requirements (handle carefully):**
- The specific file or module where `delete_task` is located

## Acceptance Criteria
- The `delete_task` test successfully covers a 404 scenario without altering the public API

## Assumptions
- The current public API is functioning correctly and should not be modified

## Related (if any)
- tests/test_tasks.py — `delete_task`
```

**Session Passport:**
- Model used: gpt-4o
- Known entities: 1
- Requirements: 1
- Unknowns flagged: 1
- Assumptions made: 1
- Preserved identifiers: `delete_task`
- Reconciled identifiers: `delete_task`
- Lexicon language: ru
- Lexicon hits: `тест, удаления, задачи`
- Lexicon mode: hybrid
- Stemmer language: ru
- Project root: /path/to/app
- ASR language: auto
- Target language: en

## Architecture

The tool is structured into core modules:

- **`core/types.py`**: IR-lite data structures (Pydantic models)
- **`core/config.py`**: Environment and API configuration
- **`core/speech.py`**: Whisper ASR wrapper + code identifier protection
- **`core/surface.py`**: Project indexing, search, and symbol cache management
- **`core/reconcile.py`**: Symbol reconciliation and fuzzy text matching
- **`core/distill.py`**: Transcript → IR-lite → prompts with identifier preservation
- **`core/prompt.py`**: Template-based prompt rendering (3 profiles)
- **`main.py`**: Typer CLI interface with file input and cache management

### Key Features

- **Project Root Support**: `--project-root` option for targeting specific projects
- **File Input**: `--file` option for reading transcripts from files
- **Profile Aliases**: `std` as shorthand for `standard`
- **Auto Language Detection**: Whisper automatically detects source language
- **Smart Translation**: LLM handles final language while preserving code
- **Safe Code Protection**: Only backticks symbols that exist in project cache
- **Symbol Cache**: Persistent inventory with AST-based extraction per project
- **Stemming Support**: Snowball stemmers for ru/es/en handle inflected forms robustly
- **Multilingual Support**: Built-in lexicons for Russian, Spanish with per-project overrides
- **Lexicon Modes**: Three processing modes (rules/llm/hybrid) for different accuracy/speed tradeoffs
- **Smart Reconciliation**: Fuzzy matching with alias generation, context rules, stemming, and LLM fallback
- **Clean Entity Display**: "Related (if any)" section shows actual paths when known, no placeholders
- **Session Tracking**: Reports project root, ASR/lexicon languages, stemmer info, processing mode, preserved/reconciled/unknown identifiers

## Development

### Running Tests

```bash
# Run all tests
uv run pytest tests/ -v

# Run specific test file
uv run pytest tests/test_basic.py -v
```

### Code Quality

The project follows Python best practices:
- Type hints throughout
- Pydantic for data validation
- Rich error handling with custom exceptions
- Modular design with clear separation of concerns
- Comprehensive docstrings

## Limitations

- Surface-level project understanding (AST used only for symbol extraction)
- Requires OpenAI API key for both ASR and distillation
- **Runtime deps**: snowballstemmer for stemming inflected forms; OpenAI API for LLM fallback
- Audio files limited to 25MB (Whisper API limit)
- Cache limited to 1000 files and 200KB per file by default; LLM candidate symbols capped at 200
- Symbol reconciliation works best with Python projects; stemming supports ru/es/en
- Cache stored under project root in `.prompt_distil/` directory
- Safe reconciliation: only symbols in cache get backticked (enforced in all modes)
- LLM mode adds API cost but handles complex/ambiguous cases better than rules alone
- Auto language mode may still produce English output due to structured JSON requirements
- No file writes outside project root's `.prompt_distil/` directory and normal build artifacts

## License

[Add your license here]

## Contributing

[Add contributing guidelines here]
