# Obsidian Meeting Notes Generator (`obsidian_notes`) — Pitfalls & Solutions Reference

Comprehensive reference of all real-world edge cases, model incompatibilities, formatting quirks, and concurrency challenges encountered during the development of the **`obsidian_notes`** extension, along with their implemented architectural solutions.

---

## 1. LLM & Agent Framework Pitfalls

### Pitfall 1: OpenAI Reasoning Models Parameter Incompatibilities (`gpt-5.6-terra`, `o1`, `o3`)

- **Symptom**: Calling `smolagents.ToolCallingAgent` configured with newer OpenAI reasoning models threw HTTP 400 bad request errors: `Unsupported parameter: 'temperature'` or failed to invoke structured tools.
- **Root Cause**: Newer OpenAI reasoning models (`o1`, `o3`, `gpt-5.*`) prohibit custom `temperature` parameters and enforce strict constraints when paired with function-calling tools via the `/v1/chat/completions` API endpoint. Additionally, tool-enabled calls require setting `reasoning_effort="none"` (or default) to prevent token exhaustion during tool argument generation.
- **Solution**:
    - In `get_smolagents_model()` within [`note_agent.py`](../note_agent.py), model names containing `terra`, `gpt-5`, `o1`, `o3`, or `reasoning` automatically inject `reasoning_effort="none"` and omit custom temperature arguments.
    - Implemented a defensive fallback interceptor: if an HTTP 400 invalid parameter error occurs during model initialization, temperature and unsupported parameters are dynamically stripped before retrying.
- **Verification**: Verified in `tests/test_smolagents_pipeline.py::test_get_smolagents_model_reasoning_effort_injection`.

---

### Pitfall 2: Brittle Markdown Scraping vs. Typed Tool Calling

- **Symptom**: Earlier prototypes prompted the LLM to output notes directly in raw Markdown or fenced JSON blocks (` ```json ... ``` `). LLMs frequently included conversational commentary (`"Sure, here is your summary:"`), unescaped backslashes, or invalid trailing commas, crashing JSON parsers.
- **Root Cause**: Generative text generation cannot enforce strict schema validation without an active function-calling feedback loop.
- **Solution**:
    - Implemented `SubmitMeetingNoteTool(Tool)` subclassing `smolagents.Tool`.
    - The agent is instructed to call `submit_meeting_note(...)` as its final step.
    - The tool handler converts inputs directly into typed Python dataclasses (`MeetingExtraction`), validating all required fields at runtime before passing the structure to `formatter.py`.
- **Verification**: Verified in `tests/test_smolagents_pipeline.py::test_submit_meeting_note_tool_execution`.

---

## 2. ASR & Subtitle Preprocessing Pitfalls

### Pitfall 3: Whisper Repetition Loops & Foreign Unicode Hallucinations

- **Symptom**: Transcripts of meetings with low audio volume, background humming, or prolonged pauses caused Whisper ASR to hallucinate infinite repetition loops (`"Thank you for watching. Thank you for watching."`) or generate non-Latin script text (Chinese Hanzi, Japanese Kana, Devanagari). When fed to the Drafter agent, the LLM hallucinated entirely fictional topics based on the foreign characters.
- **Root Cause**: Whisper decoder attention drift on out-of-domain ambient acoustic noise.
- **Solution**:
    - Built `filter_and_clean_srt()` in [`cleaner.py`](../cleaner.py).
    - Evaluates each subtitle cue against Unicode block ranges:
        - Hanzi: `[\u4e00-\u9fff]`
        - Hiragana/Katakana: `[\u3040-\u30ff]`
        - Devanagari: `[\u0900-\u097f]`
    - Detected foreign blocks are stripped or sanitized to `[UNCLEAR_AUDIO]`.
    - Verifies consecutive cue repetition: if 3 or more consecutive identical or near-identical phrases are detected, the loop is pruned.
    - Sequentially re-numbers cue indices to maintain clean SRT structure.
- **Verification**: Verified in `tests/test_cleaner.py`.

---

## 3. Action Item & Attribution Pitfalls

### Pitfall 4: Action Item Speaker Placeholders & Boilerplate Deadlines

- **Symptom**: Without explicit diarization in the audio transcript, the LLM output tasks with generic speaker placeholders (`"Unidentified speaker: Verify the configuration"`, `"Speaker 1: Review deployment"`) or useless boilerplate deadlines (`"(Due: Not stated)"`, `"Deadline: TBD"`).
- **Root Cause**: Language models trained on meeting minutes default to attributing statements to conversational agents, fabricating placeholders when specific names are absent.
- **Solution**:
    - Implemented a **Three-Layer Action Item Guarantee**:
        1. **Schema Layer**: `SubmitMeetingNoteTool` parameter docstrings strictly mandate action items begin with active imperative verbs and forbid speaker prefixes.
        2. **Agent Prompt Layer**: Drafter and Editor system instructions explicitly penalize speaker placeholders and unassigned tags.
        3. **Runtime Defensive Sanitizer (`clean_action_item_fields`)**: In [`models.py`](../models.py), regex strips colon speaker prefixes (`"Speaker 1: ..."` $\rightarrow$ `""`), extracts genuine assignees from composite tags (`"Engineer (assigned); speaker (noted)"` $\rightarrow$ `"Engineer"`), and normalizes generic labels (`owner`, `speaker`, `team`) to `"Unassigned"` and boilerplate deadlines (`not stated`, `tbd`, `n/a`) to `"None specified"`.
- **Verification**: Verified in `tests/test_models.py::test_clean_action_item_fields_*`.

---

## 4. Markdown Formatting & Obsidian Vault Pitfalls

### Pitfall 5: Table of Contents Markdown Table Column Breaking

- **Symptom**: Meeting titles containing pipe characters (`|`) broke Obsidian table rendering, causing columns to misalign and subsequent rows to be parsed incorrectly.
- **Root Cause**: The unescaped pipe character `|` in note titles directly collided with Markdown table delimiter syntax and Obsidian's internal wikilink alias format (`[[Filename|Alias]]`).
- **Solution**:
    - In [`formatter.py`](../formatter.py), `update_table_of_contents()` replaces literal pipe characters within note titles with em-dashes (`—`).
    - Markdown table wikilinks explicitly escape the alias separator using a backslash: `[[Meeting Summary - 2026-08-17 2230\|Title — Summary]]`.
    - Implemented regex row updating: existing table entries for the same note are replaced in-place, preventing duplicate table rows across re-runs.
- **Verification**: Verified in `tests/test_formatter.py::test_update_table_of_contents_pipes_and_wikilinks`.

---

### Pitfall 6: Table of Contents File Corruption on Write Interruption

- **Symptom**: If the daemon was interrupted or crashed while updating `Meeting Summaries - Table of Contents.md`, the index file was left truncated or corrupted.
- **Root Cause**: In-place overwriting without an atomic backup mechanism.
- **Solution**:
    - In `formatter.py`, `update_table_of_contents()` creates a temporary `.bak` copy of the existing TOC prior to writing updates.
    - If any exception occurs during reading, formatting, or disk write, the `.bak` file is restored over the original file before re-raising the error.
    - Upon successful write, the `.bak` file is cleanly unlinked.
- **Verification**: Verified in `tests/test_formatter.py::test_table_of_contents_atomic_rollback`.

---

### Pitfall 7: Non-Deterministic Filename Drift Across Re-runs

- **Symptom**: Re-running note generation on an existing audio file created duplicate markdown notes in the Obsidian vault because the LLM generated slightly different titles on each run (`Meeting Summary - Cloud Architecture.md` vs. `Meeting Summary - Cloud Architecture & Scaling.md`).
- **Root Cause**: Including AI-generated titles in the file path creates non-deterministic naming.
- **Solution**:
    - Decoupled note filenames from LLM titles.
    - Filenames are strictly derived from the recording timestamp:
      `Meeting Summary - YYYY-MM-DD hhmm.md`
    - Extracted using regex patterns from Samsung Voice Recorder filenames (`Voice 260817_223026.m4a` $\rightarrow$ `2026-08-17 2230`) and ISO formats.
    - Re-running the pipeline overwrites the existing note file in-place and idempotently updates its TOC row.
- **Verification**: Verified in `tests/test_formatter.py::test_extract_datetime_from_filename`.

---

## 5. Architectural & Extension Isolation Pitfalls

### Pitfall 8: Standalone Isolation vs. Monorepo Coupling

- **Symptom**: Extensions placed in separate repositories or directories failed with `ModuleNotFoundError: No module named 'steno_svc'` when tested or executed standalone.
- **Root Cause**: Hard dependency on parent package imports (`from steno_svc.extensions.base import BaseExtension`).
- **Solution**:
    - Implemented standalone fallback contracts in [`base.py`](../base.py) defining `BaseExtension` and `ExtensionContext`.
    - In [`extension.py`](../extension.py), multi-tier import guards attempt `from steno_svc.extensions.base import ...`, falling back to local `.base` or `base` definitions.
    - In [`tests/conftest.py`](../tests/conftest.py), automatic module aliasing registers the local `base` module under `sys.modules["steno_svc.extensions.base"]` if the parent service is not installed, allowing `pytest` to execute with 100% test isolation.
- **Verification**: Verified across `tests/test_extension.py` and `tests/test_smolagents_pipeline.py`.
