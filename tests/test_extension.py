from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    from obsidian_notes.extension import ObsidianNotesExtension
    from obsidian_notes.models import MeetingExtraction
    EXTENSION_MODULE = "obsidian_notes.extension"
except ImportError:
    from extension import ObsidianNotesExtension
    from models import MeetingExtraction
    EXTENSION_MODULE = "extension"

from steno_svc.extensions.base import ExtensionContext


def test_extension_default_config(tmp_path):
    ext = ObsidianNotesExtension(config_dir=tmp_path)
    assert ext.id == "obsidian_notes"
    assert ext.name == "Obsidian Meeting Notes Generator"
    assert ext.is_enabled is True
    assert "llm_model" in ext.config


def test_extension_disabled_skips(tmp_path):
    ext = ObsidianNotesExtension(config_dir=tmp_path)
    ext.set_enabled(False)
    assert ext.is_enabled is False

    ctx = ExtensionContext(
        job_id="test-job-disabled",
        filename="test.m4a",
        audio_path=tmp_path / "test.m4a",
        archive_dir=tmp_path,
        srt_path=tmp_path / "test.srt",
        srt_content="1\n00:00:00,000 --> 00:00:02,000\nHello\n",
    )
    result = ext.on_completion(ctx)
    assert result is None


def test_extension_on_completion_publishes_note(tmp_path):
    notes_dir = tmp_path / "vault_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)

    ext = ObsidianNotesExtension(config_dir=tmp_path)
    ext.config["obsidian_output_dir"] = str(notes_dir)

    fake_extraction = MeetingExtraction(
        title="Engineering Design Sync",
        meeting_focus="Discussed extension isolation.",
        summary="Discussed extension isolation.",
        tradeoffs_and_decisions=["Decouple extension tests"],
        action_items=[],
    )

    with patch(f"{EXTENSION_MODULE}.process_transcript_with_agents", return_value=fake_extraction):
        ctx = ExtensionContext(
            job_id="test-job-complete",
            filename="Voice 260817_223026.m4a",
            audio_path=tmp_path / "Voice 260817_223026.m4a",
            archive_dir=tmp_path,
            srt_path=tmp_path / "Voice 260817_223026.srt",
            srt_content="1\n00:00:00,000 --> 00:00:02,000\nDiscussion on isolation.\n",
        )
        res = ext.on_completion(ctx)
        assert res is not None
        assert "note_path" in res
        assert "artifacts" in res
        note_p = Path(res["note_path"])
        assert note_p.exists()
        assert note_p.name == "Meeting Summary - 2026-08-17 2230.md"
