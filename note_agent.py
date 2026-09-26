"""smolagents-powered distillation and final editor pipeline for standalone Obsidian notes."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from smolagents import OpenAIServerModel, Tool, ToolCallingAgent

from .models import (
    ActionItem,
    MeetingExtraction,
    Subsection,
    ThematicSection,
    sanitize_meeting_payload,
)

logger = logging.getLogger(__name__)


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


def get_smolagents_model(config: Dict[str, Any]) -> OpenAIServerModel:
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
        if any(k in model_lower for k in ("terra", "gpt-5", "o1", "o3", "reasoning")):
            model_kwargs["reasoning_effort"] = "none"

    return OpenAIServerModel(model_id=model, api_base=base_url, api_key=api_key, **model_kwargs)


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
