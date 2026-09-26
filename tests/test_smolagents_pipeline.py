"""Unit tests for smolagents-powered Drafter and Editor pipeline in obsidian_notes."""

from unittest.mock import MagicMock, patch
import pytest

try:
    from obsidian_notes.models import MeetingExtraction, ActionItem
    from obsidian_notes.note_agent import (
        SubmitMeetingNoteTool,
        get_smolagents_model,
        parse_agent_output_fallback,
        draft_note,
        edit_note,
        process_transcript_with_agents,
    )
    NOTE_AGENT_MODULE = "obsidian_notes.note_agent"
except ImportError:
    from models import MeetingExtraction, ActionItem
    from note_agent import (
        SubmitMeetingNoteTool,
        get_smolagents_model,
        parse_agent_output_fallback,
        draft_note,
        edit_note,
        process_transcript_with_agents,
    )
    NOTE_AGENT_MODULE = "note_agent"


def test_submit_meeting_note_tool():
    tool = SubmitMeetingNoteTool("meeting_sync.m4a")
    res = tool.forward(
        title="Pipeline Resilience and Architecture Tradeoffs",
        meeting_focus="The discussion centered on whether processing pipelines can support dynamic, reusable retry logic.",
        thematic_sections=[
            {
                "title": "Current Processing Engine Capabilities",
                "content": "Overview of dynamic task execution and routing.",
                "subsections": [
                    {
                        "title": "Subprocesses can supply execution data to downstream steps",
                        "content": "An isolated step runs and returns metadata to subsequent stages.",
                    }
                ],
            }
        ],
        tradeoffs_and_decisions=["Do not create and maintain monolithic static workflows"],
        action_items=[
            {"task": "Provide concrete retry implementation sample", "assignee": "AWS", "deadline": "Next week"},
            {"task": "Add identified gaps to tracking board", "assignee": "Product Lead", "deadline": "None specified"},
        ],
        side_discussions="After the meeting, the team lead guided a participant through VS Code setup.",
        closing_synthesis="The meeting established a clear direction on modularity and pipeline reliability.",
    )

    assert "successfully submitted" in res
    assert tool.extracted_data is not None
    assert isinstance(tool.extracted_data, MeetingExtraction)
    assert tool.extracted_data.title == "Pipeline Resilience and Architecture Tradeoffs"
    assert "reusable retry logic" in tool.extracted_data.meeting_focus
    assert len(tool.extracted_data.thematic_sections) == 1
    assert tool.extracted_data.thematic_sections[0].title == "Current Processing Engine Capabilities"
    assert len(tool.extracted_data.thematic_sections[0].subsections) == 1
    assert len(tool.extracted_data.action_items) == 2
    assert tool.extracted_data.action_items[0].assignee == "AWS"
    assert tool.extracted_data.side_discussions is not None
    assert "VS Code" in tool.extracted_data.side_discussions
    assert "pipeline reliability" in tool.extracted_data.closing_synthesis


def test_get_smolagents_model_missing_key_raises():
    config = {
        "llm_base_url": "https://api.openai.com/v1",
        "llm_model": "gpt-5.6-terra",
        "llm_api_key": "",
    }
    with patch.dict("os.environ", {}, clear=True):
        with pytest.raises(ValueError, match="No LLM API key configured for smolagents"):
            get_smolagents_model(config)


def test_get_smolagents_model_local_ollama():
    config = {
        "llm_base_url": "http://localhost:11434/v1",
        "llm_model": "llama3.2",
        "llm_api_key": "",
    }
    with patch.dict("os.environ", {}, clear=True):
        model = get_smolagents_model(config)
        assert model.model_id == "llama3.2"
        assert "localhost:11434" in str(model.client.base_url)


def test_parse_agent_output_fallback():
    dict_input = {
        "title": "Strategy Alignment",
        "meeting_focus": "Reviewed quarterly OKRs and migration paths.",
        "summary": "Reviewed quarterly OKRs and migration paths.",
        "tradeoffs_and_decisions": ["Focus on reliability"],
        "action_items": [],
    }
    ext = parse_agent_output_fallback(dict_input, "audio.m4a")
    assert ext.title == "Strategy Alignment"
    assert "quarterly OKRs" in ext.meeting_focus

    fence_input = '```json\n{"title": "Fence Title", "summary": "Fence summary"}\n```'
    ext2 = parse_agent_output_fallback(fence_input, "audio.m4a")
    assert ext2.title == "Fence Title"

    plain_text = "Here are the notes from the meeting."
    ext3 = parse_agent_output_fallback(plain_text, "audio.m4a")
    assert "Meeting Summary" in ext3.title
    assert "Here are the notes" in ext3.meeting_focus


def test_draft_and_edit_note_invokes_smolagents():
    with patch(f"{NOTE_AGENT_MODULE}.ToolCallingAgent") as mock_agent_cls:
        mock_agent_instance = MagicMock()
        mock_agent_cls.return_value = mock_agent_instance

        def mock_run(prompt):
            call_kwargs = mock_agent_cls.call_args[1]
            tool = call_kwargs["tools"][0]
            tool.forward(
                title="Smolagents Pipeline",
                meeting_focus="Smolagents integration completed successfully.",
                thematic_sections=[{"title": "Architecture", "content": "Detailed overview"}],
                tradeoffs_and_decisions=["Use ToolCallingAgent"],
                action_items=[{"task": "Run tests", "assignee": "DevOps", "deadline": "Now"}],
                closing_synthesis="Synthesis complete.",
            )
            return "Done"

        mock_agent_instance.run.side_effect = mock_run

        mock_model = MagicMock()
        config = {
            "llm_base_url": "https://api.openai.com/v1",
            "llm_model": "gpt-5.6-terra",
            "llm_api_key": "sk-test",
        }

        draft = draft_note("1\n00:00:01,000 --> 00:00:03,000\nHello\n", "meeting.m4a", config, model=mock_model)
        assert isinstance(draft, MeetingExtraction)
        assert draft.title == "Smolagents Pipeline"
        assert draft.meeting_focus == "Smolagents integration completed successfully."

        edited = edit_note(draft, "1\n00:00:01,000 --> 00:00:03,000\nHello\n", "meeting.m4a", config, model=mock_model)
        assert isinstance(edited, MeetingExtraction)
        assert edited.title == "Smolagents Pipeline"


def test_process_transcript_with_agents_orchestration():
    with patch(f"{NOTE_AGENT_MODULE}.get_smolagents_model") as mock_get_model, \
         patch(f"{NOTE_AGENT_MODULE}.draft_note") as mock_draft, \
         patch(f"{NOTE_AGENT_MODULE}.edit_note") as mock_edit:

        mock_model = MagicMock()
        mock_get_model.return_value = mock_model

        draft_res = MeetingExtraction(title="Draft Title", meeting_focus="Draft Focus")
        final_res = MeetingExtraction(title="Final Verified Title", meeting_focus="Final Verified Focus")

        mock_draft.return_value = draft_res
        mock_edit.return_value = final_res

        config = {"llm_api_key": "sk-test"}
        result = process_transcript_with_agents("1\n00:00:01,000 --> 00:00:02,000\nText\n", "sample.m4a", config)

        assert result == final_res
        mock_get_model.assert_called_once_with(config)
        mock_draft.assert_called_once_with("1\n00:00:01,000 --> 00:00:02,000\nText\n", "sample.m4a", config, model=mock_model)
        mock_edit.assert_called_once_with(draft_res, "1\n00:00:01,000 --> 00:00:02,000\nText\n", "sample.m4a", config, model=mock_model)
