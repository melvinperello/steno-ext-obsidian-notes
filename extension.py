"""Obsidian Notes Extension."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from steno_svc.extensions.base import BaseExtension, ExtensionContext
except ImportError:
    try:
        from .base import BaseExtension, ExtensionContext
    except (ImportError, ValueError):
        from base import BaseExtension, ExtensionContext
from .cleaner import filter_and_clean_srt
from .formatter import save_obsidian_note
from .note_agent import process_transcript_with_agents

logger = logging.getLogger(__name__)


def resolve_dir_path(raw_path: str, default_fallback: str) -> Path:
    """Expands environment variables and user profile paths."""
    if not raw_path:
        raw_path = default_fallback
    expanded = os.path.expandvars(raw_path)
    if "%USERPROFILE%" in raw_path and "%USERPROFILE%" in expanded:
        expanded = str(Path.home() / raw_path.replace("%USERPROFILE%", "").lstrip("\\/"))
    p = Path(expanded)
    if not p.is_absolute():
        p = Path.home() / "steno-svc" / p
    return p.resolve()


class ObsidianNotesExtension(BaseExtension):
    """Extension that distills audio transcripts into standalone Obsidian notes."""

    id = "obsidian_notes"
    name = "Obsidian Meeting Notes Generator"
    description = "Distills raw transcripts into structured executive notes (Drafter -> Editor pipeline) and publishes them to Obsidian with idempotent 12-hour filenames."
    enabled_by_default = True

    def __init__(self, config_dir: Optional[Path] = None) -> None:
        super().__init__(config_dir=config_dir or Path(__file__).parent)

    def default_config(self) -> Dict[str, Any]:
        return {
            "enabled": True,
            "obsidian_output_dir": "%USERPROFILE%\\steno-svc\\notes",
            "llm_base_url": "https://api.openai.com/v1",
            "llm_model": "gpt-5.6-terra",
            "llm_api_key": "",
        }

    def get_output_dir(self, context: Optional[ExtensionContext] = None) -> Path:
        env_val = os.getenv("OBSIDIAN_OUTPUT_DIR")
        if env_val and env_val.strip():
            raw = env_val.strip()
        elif self.config.get("obsidian_output_dir") or self.config.get("output_dir"):
            raw = str(self.config.get("obsidian_output_dir") or self.config.get("output_dir"))
        elif context and context.default_output_dir:
            return context.default_output_dir
        else:
            raw = "%USERPROFILE%\\steno-svc\\notes"
        return resolve_dir_path(raw, "%USERPROFILE%\\steno-svc\\notes")

    def on_completion(self, context: ExtensionContext) -> Optional[Dict[str, Any]]:
        if not self.is_enabled:
            logger.info("[obsidian_notes] Extension is disabled. Skipping note distillation.")
            return None

        output_dir = self.get_output_dir(context)
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("[obsidian_notes] Starting agentic note distillation for %s...", context.filename)

        context.update_progress(74.0, "[Obsidian Notes] Step 1/4: Subtitles cleaned (hallucination noise filtered)")
        context.log(f"Pre-processing transcript for {context.filename}...")

        # 1. Clean transcript internally before feeding to LLM
        cleaned_srt = filter_and_clean_srt(
            context.srt_content,
            strip_foreign_scripts=True,
            strip_loops=True,
        )

        # 2. Run smolagents Drafter -> Editor pipeline
        notes_data = process_transcript_with_agents(
            cleaned_srt,
            context.filename,
            self.config,
            progress_callback=context.update_progress,
            log_callback=context.log,
        )

        context.update_progress(96.0, "[Obsidian Notes] Step 3/4: Writing standalone Markdown note...")
        context.log("Formatting Obsidian note and updating Table of Contents...")

        # 3. Extract job timestamp for 12-hour format
        job_dt = datetime.now()
        if context.archive_dir:
            try:
                ts_part = context.archive_dir.name[:17]
                job_dt = datetime.strptime(ts_part, "%Y-%m-%d_%H%M%S")
            except Exception:
                try:
                    job_dt = datetime.fromtimestamp(context.audio_path.stat().st_mtime)
                except Exception:
                    job_dt = datetime.now()

        # 4. Publish standalone Obsidian markdown note
        target_note_path = save_obsidian_note(
            data=notes_data,
            srt_content=cleaned_srt,
            audio_filename=context.filename,
            output_dir=output_dir,
            created_at=job_dt,
        )

        context.update_progress(99.0, f"[Obsidian Notes] Step 4/4: Standalone note published: {target_note_path.name}")
        context.log(f"Note published: {target_note_path.name}")
        logger.info("[obsidian_notes] Note published: %s", target_note_path)
        return {
            "note_path": str(target_note_path),
            "editor_reviewed": True,
            "title": notes_data.title,
            "artifacts": [str(target_note_path)],
        }
