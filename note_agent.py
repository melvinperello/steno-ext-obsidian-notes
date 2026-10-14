"""smolagents-powered distillation and final editor pipeline for standalone Obsidian notes."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from smolagents import ChatMessage, OpenAIServerModel, Tool, ToolCallingAgent
from smolagents.models import (
    ChatMessageToolCall,
    ChatMessageToolCallFunction,
    MessageRole,
    TokenUsage,
    get_tool_json_schema,
)

from .models import (
    ActionItem,
    MeetingExtraction,
    Subsection,
    ThematicSection,
    sanitize_meeting_payload,
)

logger = logging.getLogger(__name__)


class RobustOpenAIServerModel(OpenAIServerModel):
    """OpenAIServerModel with proactive and defensive handling of OpenAI model quirks.

    - Modern OpenAI reasoning models (e.g. GPT-5, GPT-6 series like gpt-6.1-sol, o1, o3, o4)
      do not support the 'stop' parameter in /v1/chat/completions or /v1/responses.
    - Specific frontier models (such as gpt-6.1-sol) require using the Responses API
      (/v1/responses) when function/tool calling is used, because they do not support
      reasoning_effort='none' in legacy /v1/chat/completions.
    - This class automatically routes to /v1/responses when appropriate and falls back gracefully.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._unsupported_stop: Optional[bool] = None
        model_name = (self.model_id or "").split("/")[-1].lower()
        modern_no_stop_tokens = (
            "gpt-5",
            "gpt-6",
            "gpt-7",
            "o1",
            "o3",
            "o4",
            "reasoning",
            "sol",
            "astra",
            "terra",
        )
        if any(token in model_name for token in modern_no_stop_tokens):
            if model_name != "o3-mini":
                self._unsupported_stop = True

        # Models known to require the Responses API (/v1/responses) for function/tool calling
        self._use_responses_api: bool = any(tok in model_name for tok in ("sol", "astra"))

        # For models where reasoning_effort='none' is supported in chat completions (e.g. terra, o1, o3)
        if not self._use_responses_api and any(tok in model_name for tok in ("terra", "gpt-5", "o1", "o3", "reasoning")):
            if "reasoning_effort" not in self.kwargs:
                self.kwargs["reasoning_effort"] = "none"

    @property
    def supports_stop_parameter(self) -> bool:
        if self._unsupported_stop is True:
            return False
        return super().supports_stop_parameter

    def _prepare_completion_kwargs(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        res = super()._prepare_completion_kwargs(*args, **kwargs)
        if not self.supports_stop_parameter:
            res.pop("stop", None)
        return res

    def _generate_via_responses_api(
        self,
        messages: list[ChatMessage | dict],
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        """Invokes OpenAI Responses API (/v1/responses) for models requiring it with function tools."""
        input_items = []
        instructions = None

        ROLE_MAP = {
            "tool-call": "assistant",
            "tool_call": "assistant",
            "tool-response": "user",
            "tool_response": "user",
            "tool": "user",
            "function": "user",
        }

        def _normalize_content(content: Any, role: str) -> list[dict[str, Any]]:
            target_type = "output_text" if role == "assistant" else "input_text"
            if isinstance(content, str):
                return [{"type": target_type, "text": content}] if content.strip() else []
            if isinstance(content, list):
                res = []
                for item in content:
                    if isinstance(item, dict):
                        itype = item.get("type")
                        if itype in ("text", "input_text", "output_text"):
                            res.append({"type": target_type, "text": item.get("text", "")})
                        else:
                            res.append(item)
                    elif isinstance(item, str):
                        res.append({"type": target_type, "text": item})
                return res
            text = str(content)
            return [{"type": target_type, "text": text}] if text.strip() else []

        for msg in messages:
            if isinstance(msg, ChatMessage):
                raw_role = msg.role.value if hasattr(msg.role, "value") else str(msg.role)
                role = ROLE_MAP.get(raw_role, raw_role)
                if role not in ("assistant", "user", "system", "developer"):
                    role = "user"
                content = msg.content
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        fn = tc.function if hasattr(tc, "function") else tc.get("function", {})
                        func_name = fn.name if hasattr(fn, "name") else fn.get("name")
                        func_args = fn.arguments if hasattr(fn, "arguments") else fn.get("arguments")
                        if isinstance(func_args, dict):
                            func_args = json.dumps(func_args)
                        call_id = getattr(tc, "id", None) or (tc.get("id") if isinstance(tc, dict) else "call_1")
                        input_items.append({
                            "type": "function_call",
                            "call_id": call_id,
                            "name": func_name,
                            "arguments": str(func_args or "{}"),
                        })
                if role == "system":
                    if instructions is None:
                        instructions = content if isinstance(content, str) else str(content)
                    else:
                        norm_content = _normalize_content(content, role)
                        if norm_content:
                            input_items.append({"role": "system", "content": norm_content})
                else:
                    norm_content = _normalize_content(content, role)
                    if norm_content:
                        input_items.append({"role": role, "content": norm_content})
            elif isinstance(msg, dict):
                raw_role = msg.get("role", "user")
                role = ROLE_MAP.get(raw_role, raw_role)
                if role not in ("assistant", "user", "system", "developer"):
                    role = "user"
                content = msg.get("content")
                if raw_role == "tool" and msg.get("tool_call_id"):
                    input_items.append({
                        "type": "function_call_output",
                        "call_id": msg["tool_call_id"],
                        "output": str(content or ""),
                    })
                elif role == "system":
                    if instructions is None:
                        instructions = content if isinstance(content, str) else str(content)
                    else:
                        norm_content = _normalize_content(content, role)
                        if norm_content:
                            input_items.append({"role": "system", "content": norm_content})
                else:
                    norm_content = _normalize_content(content, role)
                    if norm_content:
                        input_items.append({"role": role, "content": norm_content})

        responses_tools = []
        if tools_to_call_from:
            for tool in tools_to_call_from:
                schema = get_tool_json_schema(tool)
                fn = schema.get("function", {})
                responses_tools.append({
                    "type": "function",
                    "name": fn.get("name", tool.name),
                    "description": fn.get("description", tool.description),
                    "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
                })

        req_kwargs: dict[str, Any] = {
            "model": self.model_id,
            "input": input_items,
        }
        if instructions:
            req_kwargs["instructions"] = instructions
        if responses_tools:
            req_kwargs["tools"] = responses_tools

        cfg_effort = self.kwargs.get("reasoning_effort") or kwargs.get("reasoning_effort")
        if cfg_effort and cfg_effort != "none":
            req_kwargs["reasoning"] = {"effort": cfg_effort}

        self._apply_rate_limit()
        response = self.retryer(self.client.responses.create, **req_kwargs)

        tool_calls = []
        for item in getattr(response, "output", []):
            item_type = getattr(item, "type", None) or (item.get("type") if isinstance(item, dict) else None)
            if item_type == "function_call":
                c_id = getattr(item, "call_id", None) or getattr(item, "id", None) or (item.get("call_id") if isinstance(item, dict) else "call_1")
                name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else "")
                args = getattr(item, "arguments", None) or (item.get("arguments") if isinstance(item, dict) else "{}")
                tool_calls.append(
                    ChatMessageToolCall(
                        id=c_id or "call_1",
                        type="function",
                        function=ChatMessageToolCallFunction(
                            name=name,
                            arguments=args,
                        ),
                    )
                )

        content = getattr(response, "output_text", "")
        token_usage = None
        if getattr(response, "usage", None):
            usage = response.usage
            in_tokens = getattr(usage, "prompt_tokens", 0) or getattr(usage, "input_tokens", 0)
            out_tokens = getattr(usage, "completion_tokens", 0) or getattr(usage, "output_tokens", 0)
            token_usage = TokenUsage(input_tokens=in_tokens, output_tokens=out_tokens)

        return ChatMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            tool_calls=tool_calls if tool_calls else None,
            raw=response,
            token_usage=token_usage,
        )

    def generate(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: Any,
    ) -> ChatMessage:
        if self._use_responses_api and hasattr(self.client, "responses"):
            return self._generate_via_responses_api(
                messages=messages,
                tools_to_call_from=tools_to_call_from,
                **kwargs,
            )

        try:
            return super().generate(
                messages=messages,
                stop_sequences=stop_sequences,
                response_format=response_format,
                tools_to_call_from=tools_to_call_from,
                **kwargs,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "responses" in err_msg and ("use /v1/responses" in err_msg or "use responses" in err_msg):
                logger.warning(
                    f"[obsidian_notes] Model '{self.model_id}' requires /v1/responses for function tools ({e}). "
                    "Auto-adjusting: routing to OpenAI Responses API."
                )
                self._use_responses_api = True
                return self._generate_via_responses_api(
                    messages=messages,
                    tools_to_call_from=tools_to_call_from,
                    **kwargs,
                )
            if "stop" in err_msg and ("not supported" in err_msg or "unsupported parameter" in err_msg or "unsupported_parameter" in err_msg):
                logger.warning(
                    f"[obsidian_notes] Model '{self.model_id}' does not support 'stop' parameter ({e}). "
                    "Auto-adjusting: disabling 'stop' parameter and retrying with client-side trimming."
                )
                self._unsupported_stop = True
                return super().generate(
                    messages=messages,
                    stop_sequences=stop_sequences,
                    response_format=response_format,
                    tools_to_call_from=tools_to_call_from,
                    **kwargs,
                )
            if "reasoning_effort" in err_msg:
                if any(k in err_msg for k in ("to 'none'", "to none", "function tools")) and not any(s in err_msg for s in ("does not support 'none'", "unsupported value")):
                    logger.warning(
                        f"[obsidian_notes] Model '{self.model_id}' requires reasoning_effort='none' with function tools ({e}). "
                        "Auto-adjusting: setting reasoning_effort='none' and retrying."
                    )
                    self.kwargs["reasoning_effort"] = "none"
                    kwargs["reasoning_effort"] = "none"
                    return super().generate(
                        messages=messages,
                        stop_sequences=stop_sequences,
                        response_format=response_format,
                        tools_to_call_from=tools_to_call_from,
                        **kwargs,
                    )
                else:
                    logger.warning(
                        f"[obsidian_notes] Model '{self.model_id}' rejected reasoning_effort ({e}). "
                        "Auto-adjusting: routing via Responses API if supported."
                    )
                    self.kwargs.pop("reasoning_effort", None)
                    kwargs.pop("reasoning_effort", None)
                    if hasattr(self.client, "responses"):
                        self._use_responses_api = True
                        return self._generate_via_responses_api(
                            messages=messages,
                            tools_to_call_from=tools_to_call_from,
                            **kwargs,
                        )
                    return super().generate(
                        messages=messages,
                        stop_sequences=stop_sequences,
                        response_format=response_format,
                        tools_to_call_from=tools_to_call_from,
                        **kwargs,
                    )
            raise

    def generate_stream(
        self,
        messages: list[ChatMessage | dict],
        stop_sequences: list[str] | None = None,
        response_format: dict[str, str] | None = None,
        tools_to_call_from: list[Tool] | None = None,
        **kwargs: Any,
    ):
        try:
            yield from super().generate_stream(
                messages=messages,
                stop_sequences=stop_sequences,
                response_format=response_format,
                tools_to_call_from=tools_to_call_from,
                **kwargs,
            )
        except Exception as e:
            err_msg = str(e).lower()
            if "stop" in err_msg and ("not supported" in err_msg or "unsupported parameter" in err_msg or "unsupported_parameter" in err_msg):
                logger.warning(
                    f"[obsidian_notes] Model '{self.model_id}' does not support 'stop' parameter in stream ({e}). "
                    "Auto-adjusting: disabling 'stop' parameter and retrying."
                )
                self._unsupported_stop = True
                yield from super().generate_stream(
                    messages=messages,
                    stop_sequences=stop_sequences,
                    response_format=response_format,
                    tools_to_call_from=tools_to_call_from,
                    **kwargs,
                )
            elif "reasoning_effort" in err_msg:
                if any(k in err_msg for k in ("to 'none'", "to none", "function tools")):
                    logger.warning(
                        f"[obsidian_notes] Model '{self.model_id}' requires reasoning_effort='none' in stream ({e}). "
                        "Auto-adjusting: setting reasoning_effort='none' and retrying."
                    )
                    self.kwargs["reasoning_effort"] = "none"
                    kwargs["reasoning_effort"] = "none"
                    yield from super().generate_stream(
                        messages=messages,
                        stop_sequences=stop_sequences,
                        response_format=response_format,
                        tools_to_call_from=tools_to_call_from,
                        **kwargs,
                    )
                elif "not supported" in err_msg or "unsupported" in err_msg:
                    logger.warning(
                        f"[obsidian_notes] Model '{self.model_id}' does not support 'reasoning_effort' parameter in stream ({e}). "
                        "Auto-adjusting: removing 'reasoning_effort' and retrying."
                    )
                    self.kwargs.pop("reasoning_effort", None)
                    kwargs.pop("reasoning_effort", None)
                    yield from super().generate_stream(
                        messages=messages,
                        stop_sequences=stop_sequences,
                        response_format=response_format,
                        tools_to_call_from=tools_to_call_from,
                        **kwargs,
                    )
                else:
                    raise
            else:
                raise


def clean_json_text(text: str) -> str:
    """Extract raw JSON text from markdown code fences or conversational wrappers."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text


def parse_agent_output_fallback(output: Any, filename: str) -> MeetingExtraction:
    """Defensively extracts a valid MeetingExtraction from agent string or dictionary return."""
    if isinstance(output, dict):
        return sanitize_meeting_payload(output, filename)
    text = str(output).strip()
    cleaned = clean_json_text(text)
    try:
        data = json.loads(cleaned)
        if isinstance(data, dict):
            return sanitize_meeting_payload(data, filename)
    except Exception:
        pass

    stem = Path(filename).stem.replace("_", " ").replace("-", " ").title()
    return MeetingExtraction(
        title=f"Meeting Summary: {stem}",
        meeting_focus=text[:300] if text else f"Transcribed notes from {filename}.",
        summary=text[:300] if text else f"Transcribed notes from {filename}.",
        action_items=[],
        thematic_sections=[],
        tradeoffs_and_decisions=[],
        closing_synthesis="",
    )


class SubmitMeetingNoteTool(Tool):
    """Structured tool for smolagents agents to submit deep analytical meeting facts."""

    name = "submit_meeting_note"
    description = (
        "Submits the deep analytical meeting extraction including title, meeting focus, "
        "action items, detailed thematic sections (with subsections), trade-offs/decisions, "
        "optional side discussions, and closing synthesis."
    )
    inputs = {
        "title": {
            "type": "string",
            "description": "Descriptive, professional 3-8 word title reflecting the core topic (e.g. 'Service Reliability and Architecture Tradeoffs').",
        },
        "meeting_focus": {
            "type": "string",
            "description": "Comprehensive 1-3 paragraph narrative framing the primary purpose, context, key requirements, and central dilemma.",
        },
        "action_items": {
            "type": "array",
            "description": (
                "List of actionable commitments. Each item is an object with 'task', 'assignee', and 'deadline'. "
                "CRITICAL: 'task' must start with an active imperative verb (e.g. 'Review the...', 'Configure the...'). "
                "Do NOT include speaker tags or generic prefixes like 'Unidentified speaker:' or 'Speaker:' in the task. "
                "Only set 'assignee' if a specific individual or team (e.g. 'DevOps', 'AWS') is identified; otherwise use empty string or omit. "
                "Only set 'deadline' if an explicit date, timeframe, or milestone was mentioned; NEVER output 'Not stated', 'None specified', or 'N/A'."
            ),
        },
        "thematic_sections": {
            "type": "array",
            "description": "List of in-depth thematic sections. Each section has 'title', 'content' (detailed narrative paragraphs), and optional 'subsections' (list of objects with 'title' and 'content').",
        },
        "tradeoffs_and_decisions": {
            "type": "array",
            "description": (
                "List of explicit decisions, solution paths evaluated, and consensus reached. "
                "Each entry must be a single cleanly formatted Markdown string: '**Decision / Topic**: Outcome or resolution — *Trade-off*: implications'. "
                "Do NOT output raw key-value dictionary objects (such as {'decision_or_tradeoff': ..., 'outcome': ...}). Always return strings."
            ),
        },
        "side_discussions": {
            "type": "string",
            "description": "Optional narrative of post-meeting operational support or side-tracks (e.g. IDE setup, git troubleshooting). Pass empty string or omit if none occurred.",
            "nullable": True,
        },
        "closing_synthesis": {
            "type": "string",
            "description": "Executive closing synthesis capturing the fundamental business dilemma, program dependencies, and next steps.",
            "nullable": True,
        },
    }
    output_type = "string"

    def __init__(self, filename: str) -> None:
        super().__init__()
        self.filename = filename
        self.extracted_data: Optional[MeetingExtraction] = None

    def forward(
        self,
        title: str,
        meeting_focus: str,
        thematic_sections: list[dict[str, Any]] | list[Any],
        tradeoffs_and_decisions: list[str] | list[Any],
        action_items: list[dict[str, Any]] | list[Any],
        closing_synthesis: str = "",
        side_discussions: str = "",
    ) -> str:
        payload = {
            "title": title,
            "meeting_focus": meeting_focus,
            "summary": meeting_focus,
            "thematic_sections": thematic_sections,
            "tradeoffs_and_decisions": tradeoffs_and_decisions,
            "action_items": action_items,
            "side_discussions": side_discussions.strip() if side_discussions else None,
            "closing_synthesis": closing_synthesis,
        }
        self.extracted_data = sanitize_meeting_payload(payload, self.filename)
        return "Deep analytical meeting note successfully submitted."


def get_smolagents_model(config: Dict[str, Any]) -> RobustOpenAIServerModel:
    """Returns an initialized smolagents OpenAIServerModel."""
    api_key = (
        config.get("llm_api_key")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("LLM_API_KEY")
    )
    base_url = config.get("llm_base_url") or "https://api.openai.com/v1"
    model = config.get("llm_model") or "gpt-5.6-terra"

    # Allow local servers (Ollama/LM Studio/vLLM) even without an explicit api_key
    is_local = any(h in base_url for h in ("localhost", "127.0.0.1", "0.0.0.0"))
    if not api_key:
        if is_local:
            api_key = "ollama"
        else:
            raise ValueError(
                "No LLM API key configured for smolagents. "
                "Please configure 'llm_api_key' in extension settings or set OPENAI_API_KEY."
            )

    model_kwargs: Dict[str, Any] = {}
    # OpenAI reasoning/preview models (e.g. gpt-5.6-terra, o1, o3) require reasoning_effort='none'
    # when using function calling tools via /v1/chat/completions
    cfg_reasoning = config.get("reasoning_effort")
    if cfg_reasoning:
        model_kwargs["reasoning_effort"] = cfg_reasoning
    else:
        model_lower = model.lower()
        if any(k in model_lower for k in ("terra", "gpt-5", "o1", "o3", "reasoning")) and not any(s in model_lower for s in ("sol", "astra")):
            model_kwargs["reasoning_effort"] = "none"

    return RobustOpenAIServerModel(model_id=model, api_base=base_url, api_key=api_key, **model_kwargs)


DRAFTER_INSTRUCTIONS = """You are an elite executive technical analyst and principal documentation strategist.
Your task is to analyze the provided subtitle transcript and produce an in-depth, rigorous analytical technical document.

Style and Tone Requirements (Matching Engineering Whitepaper Standard):
1. Title: Create a crisp, descriptive 3-8 word title (e.g. 'Service Reliability and Architecture Tradeoffs').
2. Meeting Focus: 1-3 thorough paragraphs framing the meeting's purpose, operational context, core dilemma, and primary questions.
3. Action Items: Extract every concrete commitment, follow-up, and task:
   - Each task MUST begin with an active imperative verb (e.g. 'Review parameter thresholds', 'Coordinate with partner teams').
   - NEVER prefix tasks or assignees with generic speaker placeholders like 'Unidentified speaker:', 'Unidentified requester liaison:', 'Speaker:', or 'Team:'.
   - Only specify an 'assignee' if an actual named individual or explicit organization/team (e.g. 'AWS', 'Infrastructure Team') is confirmed. Otherwise, leave 'assignee' empty.
   - Only specify a 'deadline' if an explicit timeline, sprint, or date is stated. NEVER write '(Due: Not stated)', 'Not specified', or 'N/A'.
4. Thematic Deep Dives: Break the discussion down into 2-5 rich thematic sections. For each theme:
   - Provide sub-sections (###) covering specific technical mechanisms, architectural patterns, and limitations.
   - Preserve exact figures, numbers, and parameters (e.g., retry thresholds, 30m/45m delays, error codes, API request formats).
   - Use bold inline typography to highlight critical concepts and takeaways.
5. Tradeoffs & Decisions: Explicitly document the solution paths evaluated and the consensus or friction points:
   - Format each item as a clean Markdown string: '**Decision / Choice**: Clear explanation of resolution or direction — *Trade-off*: Key architectural compromises or risks'.
   - NEVER output raw JSON/Python dictionaries or dictionary keys (such as `{'decision_or_tradeoff': ..., 'outcome': ...}`) in this list.
6. Post-Meeting / Side Discussions: If an operational side-track occurred after the main meeting (e.g., repository setup, tool troubleshooting, pair programming), isolate it into 'side_discussions'. If none occurred, leave it empty.
7. Closing Synthesis: A concise, impactful executive wrap-up capturing the fundamental dilemma and next milestones.

Invoke the 'submit_meeting_note' tool with your complete extraction.
"""

EDITOR_INSTRUCTIONS = """You are a meticulous senior executive editor.
Review and polish the drafted analytical note against the source SRT subtitles.

Your responsibilities:
1. Ground Truth Verification: Verify all technical claims, numbers, parameter values, and trade-offs against the transcript. Remove any assumptions or hallucinations.
2. Action Item Rigor & Clarity: Ensure all commitments are actionable, start with an active verb, and have legitimate owners only when explicitly identified. Strip any generic 'Unidentified speaker:' prefixes, 'Due: Not stated' suffixes, or unassigned boilerplate.
3. Tradeoffs & Decisions Cleanliness: Ensure all decisions are formatted as clean bold Markdown strings ('**Decision**: Outcome...'). Ensure no raw dictionary structures or key-value artifacts remain.
4. Analytical Depth & Polish: Elevate the narrative prose to be authoritative, clear, and structured. Ensure bold emphasis highlights key concepts effectively.
5. Clean Segregation: Ensure tangential post-meeting exchanges (e.g., IDE or repo setup) are cleanly isolated in 'side_discussions' rather than cluttering the core architecture sections.

Invoke the 'submit_meeting_note' tool with the verified, finalized extraction.
"""


def draft_note(
    srt_content: str,
    filename: str,
    config: Dict[str, Any],
    model: Optional[OpenAIServerModel] = None,
) -> MeetingExtraction:
    """Uses smolagents ToolCallingAgent to synthesize deep analytical note from raw SRT transcript."""
    if model is None:
        model = get_smolagents_model(config)

    submit_tool = SubmitMeetingNoteTool(filename)
    agent = ToolCallingAgent(
        tools=[submit_tool],
        model=model,
        instructions=DRAFTER_INSTRUCTIONS,
    )

    prompt = (
        f"Analyze the following meeting transcript for audio file '{filename}'.\n"
        f"Synthesize an in-depth analytical technical document and call the 'submit_meeting_note' tool.\n\n"
        f"<raw_srt_transcript>\n{srt_content}\n</raw_srt_transcript>"
    )

    result = agent.run(prompt)
    if submit_tool.extracted_data is not None:
        return submit_tool.extracted_data

    return parse_agent_output_fallback(result, filename)


def edit_note(
    draft: MeetingExtraction,
    srt_content: str,
    filename: str,
    config: Dict[str, Any],
    model: Optional[OpenAIServerModel] = None,
) -> MeetingExtraction:
    """Editor Agent: Final verification and polish step against the source transcript."""
    if model is None:
        model = get_smolagents_model(config)

    submit_tool = SubmitMeetingNoteTool(filename)
    agent = ToolCallingAgent(
        tools=[submit_tool],
        model=model,
        instructions=EDITOR_INSTRUCTIONS,
    )

    prompt = (
        f"Review and verify this drafted analytical meeting note against the source transcript for file '{filename}'.\n"
        f"Ensure strict factual accuracy, verify all numbers and action items, polish clarity, "
        f"and invoke the 'submit_meeting_note' tool with the finalized facts.\n\n"
        f"<initial_draft>\n{json.dumps(draft.to_dict(), indent=2)}\n</initial_draft>\n\n"
        f"<source_srt_transcript>\n{srt_content}\n</source_srt_transcript>"
    )

    result = agent.run(prompt)
    if submit_tool.extracted_data is not None:
        return submit_tool.extracted_data

    return parse_agent_output_fallback(result, filename)


def process_transcript_with_agents(
    srt_content: str,
    filename: str,
    config: Dict[str, Any],
    progress_callback: Optional[Any] = None,
    log_callback: Optional[Any] = None,
) -> MeetingExtraction:
    """Orchestrates the two-step smolagents ToolCallingAgent pipeline (Drafter -> Editor)."""
    model = get_smolagents_model(config)
    model_name = config.get("llm_model") or "gpt-5.6-terra"

    if progress_callback:
        progress_callback(78.0, f"[Obsidian Notes] Step 1/2: Drafter Agent synthesizing note ({model_name})...")
    if log_callback:
        log_callback(f"Step 1/2: Running smolagents Drafter Agent with model {model_name}...")

    logger.info("[obsidian_notes] Step 1: Running smolagents Drafter Agent...")
    initial_draft = draft_note(srt_content, filename, config, model=model)

    if progress_callback:
        progress_callback(90.0, f"[Obsidian Notes] Step 2/2: Editor Agent auditing claims & verifying action items ({model_name})...")
    if log_callback:
        log_callback(f"Step 2/2: Running smolagents Editor Agent for verification & polish ({model_name})...")

    logger.info("[obsidian_notes] Step 2: Running smolagents Editor Agent for verification & polish...")
    final_note = edit_note(initial_draft, srt_content, filename, config, model=model)
    return final_note
