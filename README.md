# Obsidian Meeting Notes Generator (`obsidian_notes`)

An isolated, read-only extension for [`steno-svc`](https://github.com/melvinperello/steno-svc) that transforms raw speech recognition transcripts into structured, executive-grade meeting documentation using a two-agent Hugging Face `smolagents` pipeline (Drafter $\rightarrow$ Editor) and publishes them directly to an Obsidian vault.

---

## 1. Architecture Overview

```
[steno-svc Core Daemon]
           │
           ▼ on_completion(ExtensionContext)
┌─────────────────────────────────────────────────────────────┐
│ obsidian_notes Extension Sandbox                            │
│  ├── 1. Subtitle Cleaner (cleaner.py)                       │
│  │    └── Filters Whisper loops & foreign script noise      │
│  ├── 2. smolagents Drafter Agent (note_agent.py)            │
│  │    └── Synthesizes meeting focus, themes, tradeoffs      │
│  ├── 3. smolagents Editor Agent (note_agent.py)             │
│  │    └── Fact-checks draft against ground-truth subtitles  │
│  ├── 4. Typed Tool Submission (SubmitMeetingNoteTool)       │
│  │    └── Validates dataclasses & sanitizes action items    │
│  └── 5. Markdown Formatter & Publisher (formatter.py)       │
│       ├── Idempotent 12-hour filenames (YYYY-MM-DD hhmm.md) │
│       └── Atomic Table of Contents with rollback safety     │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────┐
│ Obsidian Vault (%USERPROFILE%\steno-svc\notes\)             │
│  ├── Meeting Summary - YYYY-MM-DD hhmm.md                   │
│  └── Meeting Summaries - Table of Contents.md               │
└─────────────────────────────────────────────────────────────┘
```

---

## 2. Key Features

- **Hugging Face `smolagents` Pipeline**: Uses `ToolCallingAgent` powered by `OpenAIServerModel` in a two-agent flow (Drafter $\rightarrow$ Editor) to guarantee high factual density and eliminate hallucinations.
- **Deterministic Typed `@tool` Extraction**: Replaces fragile regex parsing of markdown blocks with a typed `@dataclass` extraction schema (`SubmitMeetingNoteTool`), validating fields at invocation time.
- **Foreign Hallucination & Repetition Defense**: Filters non-Latin script hallucinations (Chinese Hanzi, Japanese Kana, Sanskrit Devanagari) and prunes 3+ repetition loops from raw Whisper transcripts.
- **Three-Layer Action Item Sanitization**: Strips speaker colon prefixes (`"Speaker 1: Review..."` $\rightarrow$ `"Review..."`), extracts named assignees, and normalizes generic labels (`owner`, `speaker`) and boilerplate deadlines (`not stated`, `tbd`).
- **Idempotent 12-Hour Datetime Filenames**: File names derive strictly from the recording timestamp (`Meeting Summary - YYYY-MM-DD hhmm.md`), ensuring that re-running note generation updates the existing note in-place without title drift or file duplication.
- **Atomic Table of Contents**: Generates and maintains `Meeting Summaries - Table of Contents.md` sorted reverse-chronologically, with pipe-escaped wikilinks (`[[Note\|Alias]]`) and transactional `.bak` rollback safety.
- **Flexible LLM Endpoints**: Fully compatible with OpenAI (`gpt-4o`, `gpt-4o-mini`, `gpt-5.6-terra`, `o1`, `o3`), local Ollama (`http://localhost:11434/v1`), vLLM, LM Studio, or OpenRouter.

---

## 3. Project Structure

```
obsidian_notes/
├── docs/
│   ├── features.md               # Comprehensive feature and capability reference
│   └── pitfalls.md               # Detailed edge cases, quirks, and solutions reference
├── __init__.py                   # Extension package entry point
├── base.py                       # Standalone fallback contracts for isolated testing
├── cleaner.py                    # Subtitle cleaner & foreign hallucination filter
├── config.json                   # Tracked configuration template (safe generic defaults)
├── extension.py                  # BaseExtension implementation & lifecycle hook
├── formatter.py                  # Markdown renderer & Table of Contents manager
├── models.py                     # Dataclass models & regex action item sanitizers
├── note_agent.py                 # smolagents Drafter -> Editor agent pipeline
├── pyproject.toml                # Standalone package metadata & dependencies
├── secured.config.json           # Gitignored local secrets (API keys, vault path)
├── uv.lock                       # Dependency lockfile
└── tests/                        # Comprehensive unit & integration tests
    ├── conftest.py               # Vault isolation fixtures & path configuration
    ├── test_cleaner.py           # Subtitle cleaner & Unicode filter tests
    ├── test_extension.py         # Lifecycle & configuration resolution tests
    ├── test_formatter.py         # Markdown rendering & Table of Contents tests
    ├── test_models.py            # Dataclass & regex sanitizer tests
    └── test_smolagents_pipeline.py # Drafter/Editor agent pipeline tests
```

---

## 4. Configuration

The extension follows a 3-tier configuration hierarchy:

1. **In-Code Defaults**: Defined in `ObsidianNotesExtension.default_config()`.
2. **Tracked Template (`config.json`)**: Default schema values checked into version control.
3. **Secured Local Overrides (`secured.config.json`)**: Local file (strictly gitignored) storing your active API keys, custom endpoints, and vault directories.

### `config.json` Template

```json
{
    "enabled": true,
    "obsidian_output_dir": "%USERPROFILE%\\steno-svc\\notes",
    "llm_base_url": "https://api.openai.com/v1",
    "llm_model": "gpt-5.6-terra",
    "llm_api_key": ""
}
```

To configure private API keys or custom vault paths, create or edit `secured.config.json`:

```json
{
    "enabled": true,
    "obsidian_output_dir": "D:\\MyObsidianVault\\Meetings",
    "llm_base_url": "https://api.openai.com/v1",
    "llm_model": "gpt-5.6-terra",
    "llm_api_key": "YOUR_API_KEY_HERE"
}
```

Alternatively, set the `OBSIDIAN_OUTPUT_DIR` environment variable to override the destination directory at runtime.

---

## 5. Running Tests

Run the extension's self-contained test suite using `pytest`:

```powershell
uv run pytest tests/ -v
```

All 23 automated tests verify:

- Subtitle cleaning and Unicode script detection (Hanzi/Kana/Devanagari)
- Typed `SubmitMeetingNoteTool` schema validation
- LLM model initialization (OpenAI & local Ollama endpoints)
- Reasoning model parameter adaptations (`reasoning_effort="none"`)
- Drafter and Editor two-agent pipeline orchestration
- Idempotent 12-hour datetime filename extraction
- Atomic Table of Contents sorting, row updates, and rollback resilience
- Defensive action item regex sanitization

---

## 6. Detailed Documentation

- **[Features & Capabilities Reference](docs/features.md)**: Deep dive into the multi-agent pipeline, subtitle filter rules, sanitization regexes, and configuration schemas.
- **[Pitfalls & Solutions Reference](docs/pitfalls.md)**: Architectural analysis of model parameter conflicts, hallucination edge cases, markdown table pipe collisions, and standalone isolation strategies.
