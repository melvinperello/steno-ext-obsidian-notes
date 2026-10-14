"""Unit tests for smolagents-powered Drafter and Editor pipeline in obsidian_notes."""

from unittest.mock import MagicMock, patch
import pytest

try:
    from obsidian_notes.models import MeetingExtraction, ActionItem
    from obsidian_notes.note_agent import (
        RobustOpenAIServerModel,
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
        RobustOpenAIServerModel,
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


def test_robust_model_disables_stop_for_modern_openai_models():
    """Verify that new/modern OpenAI models (e.g. gpt-6.1-sol, o1) disable stop parameter."""
    for modern_model in ["gpt-6.1-sol", "gpt-6-turbo", "gpt-5.6-terra", "o1", "o3-large", "model-reasoning"]:
        cfg = {
            "llm_base_url": "https://api.openai.com/v1",
            "llm_model": modern_model,
            "llm_api_key": "sk-test",
        }
        model = get_smolagents_model(cfg)
        assert model.supports_stop_parameter is False, f"Expected {modern_model} to not support stop parameter"

        # Check that completion kwargs does NOT include 'stop'
        kwargs = model._prepare_completion_kwargs(
            messages=[{"role": "user", "content": "hi"}],
            stop_sequences=["Observation:", "Calling tools:"],
        )
        assert "stop" not in kwargs, f"'stop' should be excluded for {modern_model}"


def test_robust_model_dynamic_retry_on_unsupported_stop_error():
    """Verify that if an unknown model throws 400 unsupported parameter 'stop', it retries without stop."""
    model = RobustOpenAIServerModel(model_id="custom-unknown-model", api_key="sk-test")
    # Initially, custom model might report True for supports_stop_parameter
    call_count = 0

    def mock_super_generate(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise Exception("Error code: 400 - {'error': {'message': \"Unsupported parameter: 'stop' is not supported with this model.\", 'type': 'invalid_request_error', 'param': 'stop', 'code': 'unsupported_parameter'}}")
        from smolagents import ChatMessage
        return ChatMessage(role="assistant", content="Successfully recovered")

    with patch.object(RobustOpenAIServerModel.__bases__[0], "generate", side_effect=mock_super_generate):
        res = model.generate(
            messages=[{"role": "user", "content": "Hello"}],
            stop_sequences=["Observation:"],
        )
        assert res.content == "Successfully recovered"
        assert call_count == 2
        assert model.supports_stop_parameter is False


def test_robust_model_responses_api_for_sol():
    """Verify that gpt-6.1-sol enables Responses API and disables stop parameter without forcing reasoning_effort=none."""
    cfg = {
        "llm_base_url": "https://api.openai.com/v1",
        "llm_model": "gpt-6.1-sol",
        "llm_api_key": "sk-test",
    }
    model = get_smolagents_model(cfg)
    assert model._use_responses_api is True
    assert model.supports_stop_parameter is False
    assert model.kwargs.get("reasoning_effort") != "none"


def test_robust_model_generate_via_responses_api():
    """Verify that _generate_via_responses_api properly formats tools and extracts function calls."""
    model = RobustOpenAIServerModel(model_id="gpt-6.1-sol", api_key="sk-test")
    assert model._use_responses_api is True

    mock_resp = MagicMock()
    mock_func_item = MagicMock()
    mock_func_item.type = "function_call"
    mock_func_item.call_id = "call_abc123"
    mock_func_item.name = "submit_meeting_note"
    mock_func_item.arguments = '{"title": "Responses API Note"}'

    mock_resp.output = [mock_func_item]
    mock_resp.output_text = "I have submitted the note."
    mock_resp.usage = MagicMock(prompt_tokens=10, completion_tokens=20)

    model.client.responses.create = MagicMock(return_value=mock_resp)

    tool = SubmitMeetingNoteTool("test.m4a")
    res = model.generate(
        messages=[{"role": "user", "content": [{"type": "text", "text": "Analyze this transcript."}]}],
        tools_to_call_from=[tool],
    )

    assert res.role.value == "assistant" or str(res.role) == "assistant"
    assert res.tool_calls is not None
    assert len(res.tool_calls) == 1
    assert res.tool_calls[0].id == "call_abc123"
    assert res.tool_calls[0].function.name == "submit_meeting_note"
    assert res.tool_calls[0].function.arguments == '{"title": "Responses API Note"}'
    assert model.client.responses.create.called

    # Verify input_text type normalization
    call_kwargs = model.client.responses.create.call_args[1]
    input_items = call_kwargs["input"]
    assert input_items[0]["content"][0]["type"] == "input_text"
    assert input_items[0]["content"][0]["text"] == "Analyze this transcript."


def test_robust_model_dynamic_recovery_to_responses_api():
    """Verify that if an error asks for /v1/responses, model auto-switches to Responses API."""
    model = RobustOpenAIServerModel(model_id="custom-model", api_key="sk-test")
    model._use_responses_api = False

    def mock_super_generate(*args, **kwargs):
        raise Exception(
            "Error code: 400 - {'error': {'message': \"Function tools with reasoning_effort are not supported for "
            "custom-model in /v1/chat/completions. To use function tools, use /v1/responses or set reasoning_effort to 'none'.\", "
            "'type': 'invalid_request_error', 'param': 'reasoning_effort', 'code': None}}"
        )

    mock_resp = MagicMock()
    mock_func_item = MagicMock()
    mock_func_item.type = "function_call"
    mock_func_item.call_id = "call_fallback"
    mock_func_item.name = "submit_meeting_note"
    mock_func_item.arguments = "{}"
    mock_resp.output = [mock_func_item]
    mock_resp.output_text = ""
    mock_resp.usage = None

    model.client.responses.create = MagicMock(return_value=mock_resp)

    with patch.object(RobustOpenAIServerModel.__bases__[0], "generate", side_effect=mock_super_generate):
        res = model.generate(
            messages=[{"role": "user", "content": "Hello"}],
            tools_to_call_from=[SubmitMeetingNoteTool("test.m4a")],
        )
        assert res.tool_calls is not None
        assert res.tool_calls[0].id == "call_fallback"
        assert model._use_responses_api is True


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


def test_responses_api_multi_turn_role_mapping():
    """Verify that smolagents multi-turn messages (tool-call, tool-response) map to valid OpenAI roles."""
    from smolagents import ChatMessage
    from smolagents.models import MessageRole

    model = RobustOpenAIServerModel(model_id="gpt-6.1-sol", api_key="sk-test")
    mock_resp = MagicMock()
    mock_resp.output = []
    mock_resp.output_text = "Final confirmation"
    mock_resp.usage = None
    model.client.responses.create = MagicMock(return_value=mock_resp)

    multi_turn_messages = [
        ChatMessage(role=MessageRole.SYSTEM, content="System instructions"),
        ChatMessage(role=MessageRole.USER, content=[{"type": "text", "text": "Start analysis"}]),
        ChatMessage(
            role=MessageRole.TOOL_CALL,
            content=[{"type": "text", "text": "Calling tools:\n[{'name': 'submit_meeting_note'}]"}],
        ),
        ChatMessage(
            role=MessageRole.TOOL_RESPONSE,
            content=[{"type": "text", "text": "Observation:\nSubmitted successfully."}],
        ),
    ]

    res = model.generate(messages=multi_turn_messages)
    assert res.content == "Final confirmation"
    assert model.client.responses.create.called

    call_kwargs = model.client.responses.create.call_args[1]
    input_items = call_kwargs["input"]

    # Verify no invalid roles like 'tool-call' or 'tool-response' exist in input
    allowed_roles = {"user", "assistant", "system", "developer"}
    for item in input_items:
        if "role" in item:
            assert item["role"] in allowed_roles, f"Invalid role found: {item['role']}"

    # Specifically check that tool-call mapped to assistant with output_text
    assert input_items[1]["role"] == "assistant"
    assert input_items[1]["content"][0]["type"] == "output_text"
    assert "Calling tools:" in input_items[1]["content"][0]["text"]

    # Check that tool-response mapped to user with input_text
    assert input_items[2]["role"] == "user"
    assert input_items[2]["content"][0]["type"] == "input_text"
    assert "Observation:" in input_items[2]["content"][0]["text"]


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
