from datetime import datetime
from pathlib import Path

try:
    from obsidian_notes.models import ActionItem, MeetingExtraction, ThematicSection, Subsection
    from obsidian_notes.formatter import format_obsidian_markdown, sanitize_filename, update_table_of_contents
except ImportError:
    from models import ActionItem, MeetingExtraction, ThematicSection, Subsection
    from formatter import format_obsidian_markdown, sanitize_filename, update_table_of_contents


def test_sanitize_filename():
    assert sanitize_filename("Normal Title") == "Normal_Title"
    assert sanitize_filename("Illegal /:*? characters") == "Illegal_characters"
    assert sanitize_filename("   Multiple    Spaces   ") == "Multiple_Spaces"


def test_format_obsidian_markdown():
    data = MeetingExtraction(
        title="Weekly Sync",
        meeting_focus="Discussed service milestones and AWS architecture.",
        summary="Discussed service milestones and AWS architecture.",
        tradeoffs_and_decisions=["Migrate to NSSM", "Deploy on local machine"],
        action_items=[
            ActionItem(assignee="DevOps", task="Write tests", deadline="Today"),
            ActionItem(assignee="Unassigned", task="Verify logs", deadline="None specified"),
        ],
        thematic_sections=[
            ThematicSection(
                title="Service Architecture",
                content="Overview of service daemon mechanics.",
                subsections=[
                    Subsection(title="NSSM Wrapper", content="Running under Windows service manager.")
                ],
            )
        ],
        closing_synthesis="The migration will complete on schedule.",
    )

    srt = """1
00:00:01,000 --> 00:00:04,000
Hello, this is a test recording.

2
00:00:04,500 --> 00:00:07,000
We decided to deploy via NSSM today.
"""

    md = format_obsidian_markdown(data, srt, "test_audio.m4a")

    # Validate YAML Frontmatter
    assert "tags: [meeting-summary]" in md
    assert 'source: "test_audio.m4a"' in md
    assert "date: " in md
    assert "time: " in md

    # Validate Headers
    assert "# Weekly Sync" in md
    assert "## 1. Meeting Focus: Weekly Sync" in md
    assert "Discussed service milestones and AWS architecture." in md

    # Validate Action Items placed right after Focus
    assert "## Action Items & Next Steps" in md
    assert "- [ ] **DevOps**: Write tests *(Due: Today)*" in md
    assert "- [ ] Verify logs" in md

    # Validate Thematic Sections & Subsections
    assert "## 2. Service Architecture" in md
    assert "Overview of service daemon mechanics." in md
    assert "### 2.1 NSSM Wrapper" in md
    assert "Running under Windows service manager." in md

    # Validate Trade-Offs & Decisions
    assert "## 3. Trade-Offs & Decisions" in md
    assert "- Migrate to NSSM" in md
    assert "- Deploy on local machine" in md

    # Validate Closing Synthesis
    assert "## 4. Closing Synthesis" in md
    assert "The migration will complete on schedule." in md


def test_extract_recording_datetime():
    try:
        from obsidian_notes.formatter import extract_recording_datetime
    except ImportError:
        from formatter import extract_recording_datetime

    # Samsung Voice Recorder format
    dt = extract_recording_datetime("Voice 260817_223026.m4a")
    assert dt.year == 2026
    assert dt.month == 8
    assert dt.day == 17
    assert dt.hour == 22
    assert dt.minute == 30
    assert dt.second == 26

    # 4-digit year format
    dt2 = extract_recording_datetime("2026-09-20_14-45-00_interview.wav")
    assert dt2.year == 2026
    assert dt2.month == 9
    assert dt2.day == 20
    assert dt2.hour == 14
    assert dt2.minute == 45

    # Fallback to ingestion time
    fallback = datetime(2026, 5, 1, 9, 15, 0)
    dt3 = extract_recording_datetime("untitled_audio.m4a", fallback_dt=fallback)
    assert dt3 == fallback


def test_save_obsidian_note_idempotent(tmp_path, monkeypatch):
    try:
        from obsidian_notes.formatter import save_obsidian_note
    except ImportError:
        from formatter import save_obsidian_note

    notes_dir = tmp_path / "notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("OBSIDIAN_OUTPUT_DIR", str(notes_dir))

    data = MeetingExtraction(
        title="Sync Meeting",
        meeting_focus="Initial sync discussion.",
        summary="Initial sync discussion.",
        tradeoffs_and_decisions=["Decision 1"],
        action_items=[],
    )

    # 1. First save
    note1 = save_obsidian_note(data, "1\n00:00:01,000 --> 00:00:03,000\nText\n\n", "Voice 260817_223026.m4a")
    assert note1.exists()
    assert note1.name == "Meeting Summary - 2026-08-17 2230.md"

    toc_file = notes_dir / "Meeting Summaries - Table of Contents.md"
    assert toc_file.exists()
    toc_content = toc_file.read_text(encoding="utf-8")
    assert "[[Meeting Summary - 2026-08-17 2230\\|Sync Meeting — Initial sync discussion.]]" in toc_content

    # Notes directory has 1 note + 1 table of contents
    notes_only = [f for f in notes_dir.glob("*.md") if f.name != "Meeting Summaries - Table of Contents.md"]
    assert len(notes_only) == 1

    # 2. Second save with updated title (should overwrite note1, NOT create duplicate file)
    data_updated = MeetingExtraction(
        title="Updated Sync Meeting",
        meeting_focus="Updated summary content.",
        summary="Updated summary content.",
        tradeoffs_and_decisions=["Decision 2"],
        action_items=[],
    )
    note2 = save_obsidian_note(data_updated, "1\n00:00:01,000 --> 00:00:03,000\nText\n\n", "Voice 260817_223026.m4a")
    assert note2.exists()
    assert note2.name == "Meeting Summary - 2026-08-17 2230.md"

    # Check idempotency: exactly ONE note exists for this audio file
    notes_only = [f for f in notes_dir.glob("*.md") if f.name != "Meeting Summaries - Table of Contents.md"]
    assert len(notes_only) == 1
    assert notes_only[0] == note2

    # Check Table of Contents: updated in place without duplication and with escaped pipe
    toc_updated = toc_file.read_text(encoding="utf-8")
    assert "[[Meeting Summary - 2026-08-17 2230\\|Updated Sync Meeting — Updated summary content.]]" in toc_updated
    assert toc_updated.count("[[Meeting Summary - 2026-08-17 2230\\|") == 1
    # Check that backup file was auto-deleted upon success
    assert not (notes_dir / "Meeting Summaries - Table of Contents.md.bak").exists()


def test_table_of_contents_latest_to_oldest_order(tmp_path):
    try:
        from obsidian_notes.formatter import update_table_of_contents
    except ImportError:
        from formatter import update_table_of_contents

    toc_file = update_table_of_contents(
        note_path=tmp_path / "Meeting Summary - 2026-01-10 1000.md",
        title="January Note",
        summary_excerpt="Old note from January",
        dt=datetime(2026, 1, 10, 10, 0),
        vault_dir=tmp_path,
    )

    # Insert a future/latest note from September
    update_table_of_contents(
        note_path=tmp_path / "Meeting Summary - 2026-09-01 1400.md",
        title="September Note",
        summary_excerpt="Latest note from September",
        dt=datetime(2026, 9, 1, 14, 0),
        vault_dir=tmp_path,
    )

    # Insert a middle note from May
    update_table_of_contents(
        note_path=tmp_path / "Meeting Summary - 2026-05-15 0930.md",
        title="May Note",
        summary_excerpt="Middle note from May",
        dt=datetime(2026, 5, 15, 9, 30),
        vault_dir=tmp_path,
    )

    content = toc_file.read_text(encoding="utf-8")

    # Assert no month headers exist
    assert "## January" not in content
    assert "## September" not in content
    assert "## May" not in content

    # Assert rows are ordered latest to oldest
    idx_september = content.find("Meeting Summary - 2026-09-01 1400")
    idx_may = content.find("Meeting Summary - 2026-05-15 0930")
    idx_january = content.find("Meeting Summary - 2026-01-10 1000")

    assert idx_september != -1
    assert idx_may != -1
    assert idx_january != -1
    assert idx_september < idx_may < idx_january

    # Assert untracked headers and footer counts are omitted
    assert "This index covers" not in content
    assert "**Total recordings:**" not in content

    # Check that backup file was cleaned up on success
    bak_file = tmp_path / "Meeting Summaries - Table of Contents.md.bak"
    assert not bak_file.exists()


def test_table_of_contents_multiple_reruns_no_duplicates(tmp_path):
    try:
        from obsidian_notes.formatter import update_table_of_contents
    except ImportError:
        from formatter import update_table_of_contents

    note = tmp_path / "Meeting Summary - 2026-08-17 2101.md"

    # Run 1
    update_table_of_contents(
        note_path=note,
        title="Astra Routing | Initial Call",
        summary_excerpt="Initial discussion on routing",
        dt=datetime(2026, 8, 17, 21, 1),
        vault_dir=tmp_path,
    )
    # Run 2 with updated title
    update_table_of_contents(
        note_path=note,
        title="Astra Routing | Second Call",
        summary_excerpt="Second discussion on routing",
        dt=datetime(2026, 8, 17, 21, 1),
        vault_dir=tmp_path,
    )
    # Run 3 with third title
    toc_file = update_table_of_contents(
        note_path=note,
        title="Astra Routing | Final Call",
        summary_excerpt="Final discussion on routing",
        dt=datetime(2026, 8, 17, 21, 1),
        vault_dir=tmp_path,
    )

    content = toc_file.read_text(encoding="utf-8")
    assert content.count("Meeting Summary - 2026-08-17 2101") == 1
    assert "[[Meeting Summary - 2026-08-17 2101\\|Astra Routing — Final Call — Final discussion on routing]]" in content
    assert "Astra Routing | Final Call" not in content
    assert not (tmp_path / "Meeting Summaries - Table of Contents.md.bak").exists()


def test_table_of_contents_backup_and_rollback_on_failure(tmp_path):
    from unittest.mock import patch
    try:
        from obsidian_notes.formatter import rollback_table_of_contents, update_table_of_contents
    except ImportError:
        from formatter import rollback_table_of_contents, update_table_of_contents

    toc_file = tmp_path / "Meeting Summaries - Table of Contents.md"
    bak_file = tmp_path / "Meeting Summaries - Table of Contents.md.bak"

    # Step 1: Initial creation creates TOC (and cleans up bak on success)
    update_table_of_contents(
        note_path=tmp_path / "Meeting Summary - 2026-01-01 1000.md",
        title="Original Meeting",
        summary_excerpt="Original summary",
        dt=datetime(2026, 1, 1, 10, 0),
        vault_dir=tmp_path,
    )

    assert toc_file.exists()
    assert not bak_file.exists()
    original_text = toc_file.read_text(encoding="utf-8")

    # Step 2: Test rollback when disk write fails
    with patch.object(Path, "replace", side_effect=OSError("Simulated disk write failure")):
        try:
            update_table_of_contents(
                note_path=tmp_path / "Meeting Summary - 2026-03-01 1000.md",
                title="March Meeting",
                summary_excerpt="March summary",
                dt=datetime(2026, 3, 1, 10, 0),
                vault_dir=tmp_path,
            )
        except OSError:
            pass

    assert toc_file.read_text(encoding="utf-8") == original_text
    assert "March Meeting" not in toc_file.read_text(encoding="utf-8")

    # Step 3: Explicit rollback when bak file is present
    bak_file.write_text("CUSTOM BACKUP CONTENT", encoding="utf-8")
    assert rollback_table_of_contents(tmp_path) is True
    assert toc_file.read_text(encoding="utf-8") == "CUSTOM BACKUP CONTENT"
