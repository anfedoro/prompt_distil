# Prompt Distiller

Intent Distiller CLI - Turn noisy transcripts into clean prompts for coding agents.

## Overview

Prompt Distiller is a comprehensive tool that converts noisy voice transcripts into structured, clear prompts suitable for coding agents. It uses OpenAI's GPT-4 for intelligent parsing and Whisper for automatic speech recognition (ASR).

The tool performs intelligent project understanding with AST-based symbol extraction and generates prompts in three verbosity levels: Short, Standard, and Verbose. It features advanced status reporting, performance optimizations, and sophisticated reconciliation capabilities.

## Features

### Core Functionality
- **Speech-to-Text**: Convert audio files to text using OpenAI Whisper
- **Intelligent Distillation**: Extract structured intent from noisy transcripts
- **Multiple Output Formats**: Short, Standard, and Verbose prompts
- **Project Context**: AST-based project indexing for better context
- **Rich CLI Interface**: Beautiful console output with syntax highlighting and persistent status messages
- **Multiple Output Formats**: Rich console, Markdown, or JSON output

### Advanced Processing
- **Code Identifier Preservation**: Automatic protection of code identifiers during processing
- **Multilingual Support**: Auto language detection with intelligent translation handling
- **Symbol Cache**: Persistent symbol inventory for fast reconciliation and matching
- **Smart Reconciliation**: Fuzzy matching with stemming and LLM fallback for robust symbol mapping
- **Stemming Support**: Handles inflected forms using Snowball stemmers for ru/es/en
- **LLM Fallback**: Optional AI-assisted mapping when rule-based matching fails

### Performance & User Experience
- **Enhanced Status Reporting**: Detailed progress tracking with persistent status messages and sub-steps
- **Performance Optimizations**: Caching, timing decorators, and optimized fuzzy matching with rapidfuzz
- **Hybrid Mode Processing**: Optimized LLM-first approach with selective rule application
- **Mode-Specific Prompts**: Different system prompts for LLM-only vs hybrid/rules modes
- **Clean Backtick Handling**: Proper processing of code identifiers across different modes

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
```

Alternatively, for global use as a tool

```bash
# Install as a tool with UV
uv tool install <repository-url>
```


## Environment Variables

Prompt Distiller supports several environment variables for customizing LLM behavior:

### Model Configuration
- **`LLM_MODEL`** (default: `gpt-4o-mini`): The LLM model to use for both reconciliation and distillation
  - Examples: `gpt-4o-mini`, `gpt-4o`, `o1-preview`, `o1-mini`
  - Replaces deprecated `DISTIL_MODEL`

- **`IS_REASONING_MODEL`** (default: `false`): Set to `true` for reasoning models (o1-preview, o1-mini)
  - Values: `true`, `false`, `1`, `0`, `yes`, `no`, `on`, `off`
  - When `true`, automatically uses reasoning-compatible parameters
  - When `false`, uses temperature and standard parameters

### Response Tuning
- **`REASONING_EFFORT`** (default: `low`): Controls reasoning model effort level
  - Values: `minimal`, `low`, `medium`, `high`
  - Lower values = faster responses, higher values = more thorough reasoning

- **`VERBOSITY`** (default: `medium`): Controls response detail level
  - Values: `low`, `medium`, `high`
  - Automatically adjusted based on request type (reconciliation vs distillation)

- **`MODEL_TEMPERATURE`** (default: `0.2`): Controls response randomness for standard models
  - Range: `0.0` to `2.0`
  - Lower values = more deterministic, higher values = more creative
  - Automatically skipped for reasoning models

### Debug Settings
- **`PD_DEBUG`** (default: `0`): Enable detailed request/response logging
  - Values: `0` (disabled), `1` (enabled)

### Configuration Examples

```bash
# Standard model (GPT-4o) - fast responses
export LLM_MODEL=gpt-4o-mini
export IS_REASONING_MODEL=false
export REASONING_EFFORT=minimal
export VERBOSITY=low
export MODEL_TEMPERATURE=0.1

# Reasoning model (o1-preview) - automatic optimization
export LLM_MODEL=o1-preview
export IS_REASONING_MODEL=true
export VERBOSITY=medium
# Note: temperature automatically skipped for reasoning models

# Balanced production settings
export LLM_MODEL=gpt-4o
export IS_REASONING_MODEL=false
export REASONING_EFFORT=low
export VERBOSITY=medium
export MODEL_TEMPERATURE=0.2

# Enable debug logging
export PD_DEBUG=1
```

**Note:** Set `IS_REASONING_MODEL=true` for optimal performance with reasoning models. The system also includes automatic fallback detection for compatibility.

### Reasoning Model Limitations

When using reasoning models (like o1-preview, o1-mini):

- **Set `IS_REASONING_MODEL=true`** for optimal performance (avoids unnecessary API calls)
- **Automatic parameter handling**: `temperature` skipped, `max_completion_tokens` used instead of `max_tokens`
- **Fallback detection**: System retries with compatible parameters if flag is set incorrectly
- **Custom reasoning parameters**: Not yet supported by OpenAI API (prepared for future support)

**Automatic error handling:**
- Error 400 with `invalid_request_error` and `unsupported_value`/`unsupported_parameter`
- Parameters: `temperature`, `max_tokens`

See [docs/environment_variables.md](docs/environment_variables.md) for complete documentation.

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

# Different output profiles
uv run prompt-distil distill --text "add logging to auth flow" --profile short
uv run prompt-distil distill --text "add logging to auth flow" --profile standard
uv run prompt-distil distill --text "add logging to auth flow" --profile verbose

# Enable debug logging
uv run prompt-distil distill --text "fix the timer function" --debug

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

# Enable debug logging for audio processing
uv run prompt-distil from-audio russian_audio.wav --debug --translate

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
- **`core/lexicon.py`**: Multilingual lexicon support and language detection
- **`core/llm_map.py`**: LLM-assisted symbol mapping for complex cases
- **`core/distill.py`**: Transcript → IR-lite → prompts with identifier preservation
- **`core/prompt.py`**: Template-based prompt rendering (3 profiles)
- **`core/progress.py`**: Global progress reporting with persistent status tracking
- **`core/timing.py`**: Performance monitoring and debugging utilities
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
- **Enhanced Status Messages**: Persistent progress indicators with detailed sub-step reporting
- **Performance Monitoring**: Optional timing information with `PD_DEBUG=1` environment variable
- **Debug Logging**: Comprehensive LLM request/response analysis with `PD_DEBUG=1` or `--debug`

## Debugging and Performance

### Debug Mode
Enable detailed timing and performance information:

```bash
# Enable debug mode for performance monitoring
export PD_DEBUG=1
uv run prompt-distil distill --text "your transcript here"

# Debug mode shows timing for key operations:
# - Reconciliation processing time
# - LLM API call duration
# - Cache loading and symbol matching
# - N-gram generation and fuzzy matching
```

### Debug Logging for Reconciliation
Enable detailed debug logging for reconcile_text hybrid mode operations:

```bash
# Enable detailed debug logging via CLI flag
uv run prompt-distil distill --text "fix the login function" --debug

# Or enable via environment variable
export PD_DEBUG=1
uv run prompt-distil distill --text "fix the login function"

# Works with all commands
uv run prompt-distil from-audio recording.wav --debug
```

Debug logs are stored in `{project_root}/.prompt_distil/debug/session_YYYYMMDD_HHMMSS/` and include:
- **LLM requests and responses**: Complete prompts sent to reconciliation model and received responses
- **Symbol filtering**: Which symbols were filtered for LLM processing and why
- **N-gram comparisons**: Detailed fuzzy matching results for each n-gram against symbol aliases
- **Reconciliation summary**: Complete before/after text comparison with matched symbols
- **Error logs**: Detailed error information if reconciliation fails

Each log file is timestamped JSON with structured data for easy analysis.

### Performance Optimizations
The tool includes several performance enhancements:

- **Caching with LRU**: Lexicon loading, stemming, and alias generation are cached
- **Optimized Fuzzy Matching**: Uses rapidfuzz instead of difflib for 10x speed improvement
- **Hybrid Mode Optimization**: LLM processes only relevant symbols, then applies rules selectively
- **Symbol Processing**: All project symbols passed to LLM for maximum multilingual context
- **Bounded Scanning**: Cache limited to 1000 files and 200KB per file by default
- **Debug Logging**: Comprehensive logging system for analyzing reconciliation behavior when enabled

### Status Reporting Features
Enhanced progress tracking shows:
- Main process steps with persistent checkmarks
- Detailed sub-steps for reconciliation model calls
- N-gram search progress indicators
- Explicit reporting of LLM API calls
- Backtick cleaning confirmation
- Model response parsing status
- Debug logging session information when enabled

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
- **Runtime deps**: snowballstemmer for stemming inflected forms; OpenAI API for LLM fallback; rapidfuzz for optimized fuzzy matching
- Audio files limited to 25MB (Whisper API limit)
- Cache limited to 1000 files and 200KB per file by default; LLM processes all project symbols for maximum multilingual context
- Symbol reconciliation works best with Python projects; stemming supports ru/es/en
- Cache stored under project root in `.prompt_distil/` directory
- Safe reconciliation: only symbols in cache get backticked (enforced in all modes)
- LLM mode adds API cost but handles complex/ambiguous cases better than rules alone
- Auto language mode may still produce English output due to structured JSON requirements
- No file writes outside project root's `.prompt_distil/` directory and normal build artifacts

## Roadmap

### Upcoming Features

#### Microphone Integration
- **Real-time Audio Capture**: Direct microphone input for live transcription
- **Streaming ASR**: Process audio as you speak for faster feedback
- **Voice Activity Detection**: Automatic start/stop recording based on speech detection
- **Multiple Audio Devices**: Support for different microphone inputs and configurations
- **Noise Reduction**: Built-in audio preprocessing for cleaner transcripts

#### ASR Model Integration
- **Local ASR Support**: Integration with local speech recognition models for privacy
- **Custom Model Training**: Support for domain-specific ASR model fine-tuning
- **Offline Processing**: Ability to process audio without internet connectivity
- **Multiple ASR Backends**: Choice between OpenAI Whisper, local models, and other providers
- **Language Model Optimization**: Better handling of technical terminology and code-related speech

#### Enhanced Project Understanding
- **Deeper AST Analysis**: More sophisticated code structure understanding
- **Cross-language Support**: Better support for JavaScript, TypeScript, Go, Rust projects
- **API Documentation Integration**: Automatic inclusion of API docs and schemas in context
- **Git Integration**: Incorporate recent changes and commit history into prompts
- **Dependency Analysis**: Understanding of project dependencies and their APIs

#### Advanced Features
- **Interactive Mode**: Step-by-step refinement of generated prompts
- **Template System**: Custom prompt templates for different types of tasks
- **Integration APIs**: REST API for programmatic access to distillation functionality
- **Batch Processing**: Process multiple audio files or transcripts simultaneously
- **Configuration Profiles**: Saved configurations for different projects and use cases

### Contributing to the Roadmap
We welcome feedback and contributions! If you have ideas for features or would like to contribute to development, please open an issue or submit a pull request.

## License

MIT License

Copyright (c) 2024 Prompt Distiller

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Contributing

[Add contributing guidelines here]
