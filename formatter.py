"""Obsidian-compatible Markdown formatter for deep analytical voice notes and Table of Contents indexing."""

from __future__ import annotations

import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import MeetingExtraction, clean_action_item_fields

logger = logging.getLogger(__name__)


def sanitize_filename(title: str) -> str:
    """Cleans note title for safe filesystem filenames."""
    cleaned = re.sub(r'[\\/*?:"<>|]', "", title)
    cleaned = re.sub(r"\s+", "_", cleaned.strip())
    return cleaned[:60] or "Voice_Note"


def extract_recording_datetime(filename: str, fallback_dt: Optional[datetime] = None) -> datetime:
    """Extracts recording date and time from audio filename patterns (e.g. 'Voice 260817_223026.m4a').

    Supports:
    - 4-digit year: YYYYMMDD_HHMMSS, YYYY-MM-DD_HH-MM-SS, etc.
    - 2-digit year: Voice YYMMDD_HHMMSS (Samsung Voice Recorder)
    - Fallback: fallback_dt or datetime.now() (ingestion time)
    """
    m = re.search(r"(?:^|[^\d])(20\d\d)[-_]?([01]\d)[-_]?([0-3]\d)[_T\s-]+([0-2]\d)[-_.:]?([0-5]\d)(?:[-_.:]?([0-5]\d))?", filename)
    if m:
        try:
            y, mo, d, h, mi, s = m.groups()
            return datetime(int(y), int(mo), int(d), int(h), int(mi), int(s or 0))
        except ValueError:
            pass

    m2 = re.search(r"(?:Voice[_\s-]*)?(\d{2})([01]\d)([0-3]\d)[_T\s-]+([0-2]\d)([0-5]\d)([0-5]\d)", filename, re.IGNORECASE)
    if m2:
        try:
            y, mo, d, h, mi, s = m2.groups()
            return datetime(2000 + int(y), int(mo), int(d), int(h), int(mi), int(s))
        except ValueError:
            pass

    return fallback_dt or datetime.now()


def format_obsidian_markdown(
    data: MeetingExtraction,
    srt_content: str,
    audio_filename: str,
    created_at: Optional[datetime] = None,
) -> str:
    """Assembles a deep analytical Obsidian note matching the cd_notes vault standard.

    Structure:
    1. Vault YAML Frontmatter (date, time, source, tags: [meeting-summary])
    2. # Title
    3. ## 1. Meeting Focus: [Title / Topic]
    4. ## Action Items & Next Steps (Placed right after Focus if any exist)
    5. ## 2... Thematic Deep Dives (with ### subsections, bold emphasis, parameters)
    6. ## Trade-Offs & Decisions (if present)
    7. ## Post-Meeting Technical Support & Side Discussions (if present)
    8. ## Closing Synthesis (if present)
    """
    dt = created_at or extract_recording_datetime(audio_filename)
    date_iso = dt.strftime("%Y-%m-%d")
    time_24h = dt.strftime("%H:%M")

    # 1. Frontmatter
    frontmatter = f"""---
date: {date_iso}
time: {time_24h}
source: "{audio_filename}"
source_audio: "{audio_filename}"
tags: [meeting-summary]
---"""

    parts = [frontmatter, "", f"# {data.title}", ""]

    # 2. Meeting Focus
    parts.append(f"## 1. Meeting Focus: {data.title}")
    parts.append("")
    focus_text = data.meeting_focus or data.summary or "Voice recording transcribed and processed locally."
    parts.append(focus_text)
    parts.append("")

    # 3. Action Items immediately after Focus (per user preference)
    if data.action_items:
        parts.append("## Action Items & Next Steps")
        parts.append("")
        for a in data.action_items:
            clean_item = clean_action_item_fields(task=a.task, assignee=a.assignee, deadline=a.deadline)
            task = clean_item.task
            if task and task[0].islower():
                task = task[0].upper() + task[1:]

            deadline_str = ""
            if clean_item.deadline and clean_item.deadline.lower() not in (
                "none specified", "not specified", "not stated", "none", "n/a", "unknown"
            ):
                deadline_str = f" *(Due: {clean_item.deadline})*"

            assignee_clean = clean_item.assignee.strip() if clean_item.assignee else ""
            is_generic = any(
                w in assignee_clean.lower()
                for w in ("unidentified", "unknown", "unassigned", "speaker", "liaison", "stakeholder", "owner", "team", "parties", "none")
            )
            assignee_str = f"**{assignee_clean}**: " if assignee_clean and not is_generic else ""
            parts.append(f"- [ ] {assignee_str}{task}{deadline_str}")
        parts.append("")

    # 4. Thematic Deep Dives
    section_counter = 2
    if data.thematic_sections:
        for sec in data.thematic_sections:
            sec_header = f"## {section_counter}. {sec.title}" if sec.title else f"## {section_counter}. Discussion Details"
            parts.append(sec_header)
            parts.append("")
            if sec.content:
                parts.append(sec.content)
                parts.append("")
            if sec.subsections:
                sub_counter = 1
                for sub in sec.subsections:
                    sub_title = sub.title or f"Point {sub_counter}"
                    parts.append(f"### {section_counter}.{sub_counter} {sub_title}")
                    parts.append("")
                    if sub.content:
                        parts.append(sub.content)
                        parts.append("")
                    sub_counter += 1
            section_counter += 1

    # 5. Trade-Offs and Decisions
    decisions = data.tradeoffs_and_decisions or data.key_decisions
    if decisions:
        from .models import sanitize_tradeoffs_and_decisions
        cleaned_decisions = sanitize_tradeoffs_and_decisions(decisions)
        if cleaned_decisions:
            parts.append(f"## {section_counter}. Trade-Offs & Decisions")
            parts.append("")
            for d in cleaned_decisions:
                parts.append(f"- {d}")
            parts.append("")
            section_counter += 1

    # 6. Post-Meeting / Side Discussions (if any)
    if data.side_discussions:
        parts.append(f"## {section_counter}. Post-Meeting Technical Support & Side Discussions")
        parts.append("")
        parts.append(data.side_discussions)
        parts.append("")
        section_counter += 1

    # 7. Closing Synthesis
    if data.closing_synthesis:
        parts.append(f"## {section_counter}. Closing Synthesis")
        parts.append("")
        parts.append(data.closing_synthesis)
        parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def rollback_table_of_contents(vault_dir: Path) -> bool:
    """Restores 'Meeting Summaries - Table of Contents.md' from its .bak backup file if available."""
    toc_path = vault_dir / "Meeting Summaries - Table of Contents.md"
    bak_path = vault_dir / "Meeting Summaries - Table of Contents.md.bak"
    if bak_path.exists():
        try:
            shutil.copy2(bak_path, toc_path)
            logger.info("Successfully rolled back Table of Contents from %s", bak_path.name)
            return True
        except Exception as exc:
            logger.error("Failed to rollback Table of Contents from backup: %s", exc)
            return False
    return False


def _sanitize_row_pipes(line: str) -> str:
    """Ensures wikilink aliases inside markdown table cells have their pipe escaped as \\|,
    and replaces any additional raw pipes in the title/summary with an em-dash (—).
    """
    def _fix_wikilink(m: re.Match) -> str:
        raw = m.group(1)
        normalized = raw.replace(r"\|", "|")
        if "|" in normalized:
            parts = normalized.split("|", 1)
            target = parts[0].strip()
            display = parts[1].replace("|", "—").strip()
            return f"[[{target}\\|{display}]]"
        return f"[[{raw}]]"

    return re.sub(r"\[\[(.*?)\]\]", _fix_wikilink, line)


def update_table_of_contents(
    note_path: Path,
    title: str,
    summary_excerpt: str,
    dt: datetime,
    vault_dir: Path,
    stale_stems: Optional[list[str]] = None,
) -> Path:
    """Maintains 'Meeting Summaries - Table of Contents.md' in the vault directory.
    Inserts or updates the meeting row in a single unified table arranged latest to oldest.
    Maintains a safe .bak backup before writing and automatically rolls back on any error.
    Cleans up the .bak backup upon successful atomic write.
    Guarantees no duplicate entries even if rerun multiple times.
    """
    toc_path = vault_dir / "Meeting Summaries - Table of Contents.md"
    bak_path = vault_dir / "Meeting Summaries - Table of Contents.md.bak"

    # Sanitize title and summary so internal pipes don't break table columns
    clean_title = title.replace("|", "—").strip()
    clean_summary = summary_excerpt.replace("\n", " ").replace("|", "—").strip()
    if len(clean_summary) > 120:
        clean_summary = clean_summary[:117] + "..."

    date_str = dt.strftime("%b %d, %Y – %I:%M %p").replace(" 0", " ")
    # Wikilink alias delimiter inside markdown table must be escaped as \|
    table_row = f"| {date_str} | [[{note_path.stem}\\|{clean_title} — {clean_summary}]] |"
    note_stem = note_path.stem
    new_sort_key = dt.strftime("%Y-%m-%d %H%M")

    if not toc_path.exists():
        initial_content = f"""---
tags: [table-of-contents, meeting-summaries]
created: {datetime.now().strftime("%Y-%m-%d")}
---

# 📋 Meeting Summaries — Table of Contents

| Date & Time | Summary |
|---|---|
{table_row}
"""
        toc_path.write_text(initial_content, encoding="utf-8")
        return toc_path

    # Read original content and create immediate disk backup before mutation
    original_content = toc_path.read_text(encoding="utf-8")
    try:
        bak_path.write_text(original_content, encoding="utf-8")
    except Exception as exc:
        logger.warning("Could not create Table of Contents backup file: %s", exc)

    try:
        lines = original_content.splitlines()

        header_lines = []
        existing_rows = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("|"):
                in_table = True
                is_row = (
                    not stripped.lower().startswith("| date")
                    and not re.match(r"^\|[-| :]+\|$", stripped)
                )
                if is_row:
                    existing_rows.append(stripped)
            elif not in_table:
                # Lines strictly before table (frontmatter, document title)
                if (
                    not stripped.startswith("## ")
                    and not stripped.startswith("This index covers")
                    and not stripped.startswith("> **Total recordings:**")
                    and not stripped.startswith("> Generated automatically")
                ):
                    header_lines.append(line)

        header_text = "\n".join(header_lines).rstrip()
        header_text += "\n\n| Date & Time | Summary |\n|---|---|"

        # Process and sort rows latest to oldest with deduplication
        parsed_entries = []
        seen_targets: set[str] = set()
        replaced = False

        # Identify all stems that belong to this note (canonical stem + any stale stems)
        matching_stems = {note_stem}
        if stale_stems:
            matching_stems.update(stale_stems)

        for r in existing_rows:
            target_match = re.search(r"\[\[([^\\|\]]+)", r)
            target_stem = target_match.group(1).strip() if target_match else None

            if target_stem and target_stem in matching_stems:
                if not replaced:
                    parsed_entries.append((new_sort_key, table_row))
                    seen_targets.add(note_stem)
                    replaced = True
                # Skip duplicate rows pointing to this note or its stale variants
            else:
                if target_stem:
                    if target_stem in seen_targets:
                        # Drop duplicate entries pointing to the same target note
                        continue
                    seen_targets.add(target_stem)

                fixed_r = _sanitize_row_pipes(r)
                m = re.search(r"\[\[Meeting Summary - (\d{4}-\d{2}-\d{2} \d{4})", fixed_r)
                sort_key = m.group(1) if m else "0000-00-00 0000"
                parsed_entries.append((sort_key, fixed_r))

        if not replaced:
            parsed_entries.append((new_sort_key, table_row))
            seen_targets.add(note_stem)

        # Sort descending: latest to oldest
        parsed_entries.sort(key=lambda x: x[0], reverse=True)

        new_content = header_text + "\n" + "\n".join(e[1] for e in parsed_entries) + "\n"

        # Atomic write: write to temp file first, then atomically replace
        temp_toc = vault_dir / f".tmp_toc_{os.getpid()}_{int(datetime.now().timestamp() * 1000)}.md"
        temp_toc.write_text(new_content, encoding="utf-8")
        temp_toc.replace(toc_path)

        # Auto-delete backup file upon successful atomic write to keep vault clean
        if bak_path.exists():
            try:
                bak_path.unlink()
                logger.debug("Cleaned up temporary TOC backup: %s", bak_path.name)
            except Exception as exc:
                logger.warning("Could not remove TOC backup file: %s", exc)

        return toc_path

    except Exception as exc:
        logger.error("Error while updating Table of Contents, rolling back: %s", exc)
        try:
            if original_content is not None:
                toc_path.write_text(original_content, encoding="utf-8")
            elif bak_path.exists():
                shutil.copy2(bak_path, toc_path)
        except Exception as rollback_err:
            logger.critical("Rollback failed: %s", rollback_err)
        raise


def save_obsidian_note(
    data: MeetingExtraction,
    srt_content: str,
    audio_filename: str,
    output_dir: Optional[Path] = None,
    created_at: Optional[datetime] = None,
) -> Path:
    """Formats and writes the standalone Obsidian note and updates the vault Table of Contents.

    Naming convention: 'Meeting Summary - YYYY-MM-DD HHMM.md'
    """
    if output_dir is None:
        env_val = os.getenv("OBSIDIAN_OUTPUT_DIR")
        if env_val and env_val.strip():
            output_dir = Path(os.path.expandvars(env_val.strip()))
        else:
            output_dir = Path(os.path.expandvars(r"%USERPROFILE%\steno-svc\notes"))

    output_dir.mkdir(parents=True, exist_ok=True)

    dt = extract_recording_datetime(audio_filename, fallback_dt=created_at)
    date_str = dt.strftime("%Y-%m-%d")
    time_str_24h = dt.strftime("%H%M")
    canonical_filename = f"Meeting Summary - {date_str} {time_str_24h}.md"
    target_path = output_dir / canonical_filename

    # Disambiguate if collision occurs with a different audio file
    if target_path.exists():
        try:
            sample = target_path.read_text(encoding="utf-8")[:1000]
            if f'source: "{audio_filename}"' not in sample and f'source_audio: "{audio_filename}"' not in sample:
                audio_stem = Path(audio_filename).stem
                canonical_filename = f"Meeting Summary - {date_str} {time_str_24h}_{sanitize_filename(audio_stem)}.md"
                target_path = output_dir / canonical_filename
        except Exception:
            pass

    # Scan for existing notes referencing this source audio file
    existing_notes_for_audio: list[Path] = []
    audio_stem = Path(audio_filename).stem
    for note_file in output_dir.glob("*.md"):
        if note_file.name == "Meeting Summaries - Table of Contents.md":
            continue
        try:
            content_sample = note_file.read_text(encoding="utf-8")[:1000]
            if (f'source: "{audio_filename}"' in content_sample
                or f'source_audio: "{audio_filename}"' in content_sample
                or f"_{audio_stem}" in note_file.stem):
                existing_notes_for_audio.append(note_file)
        except Exception:
            pass

    content = format_obsidian_markdown(data, srt_content, audio_filename, created_at=dt)
    target_path.write_text(content, encoding="utf-8")

    # Clean up stale notes for this same audio file to preserve idempotency
    for old_note in existing_notes_for_audio:
        if old_note.resolve() != target_path.resolve() and old_note.exists():
            try:
                old_note.unlink()
            except Exception:
                pass

    # Maintain vault Table of Contents
    try:
        excerpt = data.meeting_focus or data.summary or ""
        stale_stems = [
            old_note.stem
            for old_note in existing_notes_for_audio
            if old_note.resolve() != target_path.resolve()
        ]
        update_table_of_contents(
            target_path,
            data.title,
            excerpt,
            dt,
            output_dir,
            stale_stems=stale_stems,
        )
    except Exception as exc:
        logger.error("Failed to update Table of Contents: %s", exc)

    return target_path
