"""Data models and deterministic sanitizers for the Obsidian Notes extension.
Follows the mp-wonderful-world Deterministic Sandwich Architecture:
- Zero external validation dependencies (dataclasses only).
- Defensive fallback handling (never crashes on unexpected LLM shapes).
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ActionItem:
    """Concrete actionable task extracted from voice recording."""
    task: str = "Review recording takeaways"
    assignee: str = "Unassigned"
    deadline: str = "None specified"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class Subsection:
    """A subsection within a thematic deep-dive (rendered as ### X.Y)."""
    title: str = ""
    content: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ThematicSection:
    """A major thematic deep-dive section (rendered as ## X)."""
    title: str = ""
    content: str = ""
    subsections: List[Subsection] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MeetingExtraction:
    """Comprehensive meeting extraction distilled by smolagents."""
    title: str = "Voice Note Summary"
    meeting_focus: str = "Voice recording transcribed and processed locally."
    summary: str = "Voice recording transcribed and processed locally."
    action_items: List[ActionItem] = field(default_factory=list)
    thematic_sections: List[ThematicSection] = field(default_factory=list)
    tradeoffs_and_decisions: List[str] = field(default_factory=list)
    side_discussions: Optional[str] = None
    closing_synthesis: str = ""

    # Backward-compatibility aliases
    key_decisions: List[str] = field(default_factory=list)
    topics_discussed: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


GENERIC_ASSIGNEE_PATTERNS = (
    "unidentified",
    "unknown",
    "unassigned",
    "speaker",
    "liaison",
    "stakeholder",
    "owner",
    "team",
    "parties",
    "none",
    "n/a",
    "not stated",
    "not specified",
)

GENERIC_DEADLINE_PATTERNS = (
    "none specified",
    "not specified",
    "not stated",
    "none",
    "n/a",
    "unknown",
    "not given",
    "tbd",
)


def clean_action_item_fields(task: str, assignee: str = "", deadline: str = "") -> ActionItem:
    """Sanitizes task description, assignee, and deadline to ensure clean actionable items."""
    task = task.strip()
    assignee = assignee.strip()
    deadline = deadline.strip()

    # 1. Clean trailing (Due: ...) boilerplate from task if present
    task = re.sub(
        r"\s*\((?:due|deadline):\s*(?:not stated|not specified|none|none specified|n/a|unknown)\)\.?$",
        "",
        task,
        flags=re.IGNORECASE,
    ).strip()

    # 2. Clean leading generic speaker/assignee prefixes from task
    # Case A: Named assignee with unidentified follow-up, e.g. "Assignee (assignment); unidentified speaker (follow-up): Assign..."
    match_named_with_unidentified = re.match(
        r"^([A-Za-z0-9_\-\s]+)\s*\([^)]*\)\s*;\s*unidentified[^:]*:\s*(.+)$",
        task,
        re.IGNORECASE,
    )
    if match_named_with_unidentified:
        if not assignee or any(g in assignee.lower() for g in ("unidentified", "unknown", "unassigned", "speaker")):
            assignee = match_named_with_unidentified.group(1).strip()
        task = match_named_with_unidentified.group(2).strip()
    else:
        # Case B: Generic speaker prefix: "Unidentified speaker:", "Unidentified requester liaison:", "Speaker:", etc.
        match_generic = re.match(
            r"^(?:(?:unidentified|unknown|unassigned|various)\s+[^:]*|speaker(?: \d+)?|attendee|liaison):\s*(.+)$",
            task,
            re.IGNORECASE,
        )
        if match_generic:
            task = match_generic.group(1).strip()
        else:
            # Case C: Colon prefix where prefix might be a speaker or assignee
            colon_match = re.match(r"^([A-Za-z0-9_\-\s/]+):\s*(.+)$", task)
            if colon_match and len(colon_match.group(1)) <= 35:
                prefix = colon_match.group(1).strip()
                if any(g in prefix.lower() for g in ("unidentified", "unknown", "unassigned", "speaker", "liaison")):
                    task = colon_match.group(2).strip()
                elif not assignee or assignee.lower() in ("unassigned", "none", "unknown"):
                    assignee = prefix
                    task = colon_match.group(2).strip()

    # 3. Capitalize first letter of task
    if task and task[0].islower():
        task = task[0].upper() + task[1:]

    # 4. Clean generic assignee
    if any(g in assignee.lower() for g in GENERIC_ASSIGNEE_PATTERNS):
        assignee = "Unassigned"

    # 5. Clean generic deadline
    if any(d in deadline.lower() for d in GENERIC_DEADLINE_PATTERNS) or deadline.lower().startswith("due:"):
        deadline = "None specified"

    return ActionItem(
        task=task or "Review meeting takeaways",
        assignee=assignee or "Unassigned",
        deadline=deadline or "None specified",
    )


def sanitize_action_items(raw_items: Any) -> List[ActionItem]:
    """Defensively parses action items from any arbitrary LLM shape."""
    items: List[ActionItem] = []
    if not isinstance(raw_items, list):
        return items

    for item in raw_items:
        if isinstance(item, dict):
            task = str(item.get("task", "")).strip()
            if task:
                items.append(
                    clean_action_item_fields(
                        task=task,
                        assignee=str(item.get("assignee", "")).strip(),
                        deadline=str(item.get("deadline", "")).strip(),
                    )
                )
        elif isinstance(item, str) and item.strip():
            items.append(clean_action_item_fields(task=item.strip()))

    return items


def sanitize_string_list(raw_list: Any) -> List[str]:
    """Defensively coerces a list of strings from arbitrary input."""
    if not isinstance(raw_list, list):
        if isinstance(raw_list, str) and raw_list.strip():
            return [raw_list.strip()]
        return []
    return [str(x).strip() for x in raw_list if str(x).strip()]


def sanitize_tradeoffs_and_decisions(raw_decisions: Any) -> List[str]:
    """Defensively parses decisions and trade-offs into clean, human-readable Markdown bullet strings.
    Gracefully handles plain strings, dictionaries (e.g. {'decision': ..., 'tradeoff': ...}),
    or stringified dictionary representations emitted by LLMs.
    """
    if not isinstance(raw_decisions, list):
        if isinstance(raw_decisions, (str, dict)):
            raw_decisions = [raw_decisions]
        else:
            return []

    results: List[str] = []
    for item in raw_decisions:
        if not item:
            continue

        # If stringified dict literal, e.g. "{'decision_or_tradeoff': ..., 'outcome': ...}"
        if isinstance(item, str) and item.strip().startswith("{") and item.strip().endswith("}"):
            try:
                import ast
                evaluated = ast.literal_eval(item.strip())
                if isinstance(evaluated, dict):
                    item = evaluated
            except Exception:
                pass

        if isinstance(item, dict):
            # Form 1: decision_or_tradeoff + outcome
            if "decision_or_tradeoff" in item and "outcome" in item:
                title = str(item.get("decision_or_tradeoff", "")).strip()
                outcome = str(item.get("outcome", "")).strip()
                if title and outcome:
                    results.append(f"**{title}**: {outcome}")
                elif title or outcome:
                    results.append(f"**{title or outcome}**")
                continue

            # Form 2: decision + tradeoff
            if "decision" in item and "tradeoff" in item:
                dec = str(item.get("decision", "")).strip()
                tro = str(item.get("tradeoff", "")).strip()
                if dec and tro:
                    results.append(f"**{dec}** — *Trade-off*: {tro}")
                elif dec or tro:
                    results.append(f"**{dec or tro}**")
                continue

            # Form 3: decision + rationale / resolution
            if "decision" in item and ("rationale" in item or "resolution" in item or "outcome" in item):
                dec = str(item.get("decision", "")).strip()
                res = str(item.get("rationale") or item.get("resolution") or item.get("outcome", "")).strip()
                if dec and res:
                    results.append(f"**{dec}**: {res}")
                elif dec or res:
                    results.append(f"**{dec or res}**")
                continue

            # Form 4: generic dict fallback
            parts = []
            for k, v in item.items():
                if v and str(v).strip():
                    clean_k = k.replace("_", " ").title()
                    parts.append(f"**{clean_k}**: {str(v).strip()}")
            if parts:
                results.append(" — ".join(parts))
        elif isinstance(item, str) and item.strip():
            results.append(item.strip())
        else:
            s = str(item).strip()
            if s:
                results.append(s)

    return results


def sanitize_thematic_sections(raw_sections: Any) -> List[ThematicSection]:
    """Defensively parses thematic sections and their subsections."""
    sections: List[ThematicSection] = []
    if not isinstance(raw_sections, list):
        return sections

    for s in raw_sections:
        if isinstance(s, dict):
            sec_title = str(s.get("title", "")).strip()
            sec_content = str(s.get("content", "")).strip()
            raw_sub = s.get("subsections", [])
            subsections: List[Subsection] = []
            if isinstance(raw_sub, list):
                for sub in raw_sub:
                    if isinstance(sub, dict):
                        sub_title = str(sub.get("title", "")).strip()
                        sub_content = str(sub.get("content", "")).strip()
                        if sub_title or sub_content:
                            subsections.append(Subsection(title=sub_title, content=sub_content))
                    elif isinstance(sub, str) and sub.strip():
                        subsections.append(Subsection(title="", content=sub.strip()))
            if sec_title or sec_content or subsections:
                sections.append(ThematicSection(title=sec_title, content=sec_content, subsections=subsections))
        elif isinstance(s, str) and s.strip():
            sections.append(ThematicSection(title="Discussion Details", content=s.strip()))

    return sections


def sanitize_meeting_payload(raw: Any, filename: str = "audio.m4a") -> MeetingExtraction:
    """Sanitizes raw JSON or tool payload into a strictly typed MeetingExtraction dataclass.
    Guarantees that missing, malformed, or unexpected keys never crash the pipeline.
    """
    if not isinstance(raw, dict):
        raw = {}

    # Extract title
    raw_title = str(raw.get("title", "")).strip()
    if not raw_title:
        stem = Path(filename).stem.replace("_", " ").replace("-", " ").title()
        raw_title = f"Meeting Summary: {stem}"

    # Extract focus and summary
    focus = str(raw.get("meeting_focus") or raw.get("summary") or "").strip()
    if not focus:
        focus = f"Voice recording transcribed from {filename}."

    # Action items
    actions = sanitize_action_items(raw.get("action_items", []))

    # Thematic sections
    thematic_sections = sanitize_thematic_sections(raw.get("thematic_sections", []))

    # Decisions and trade-offs
    decisions = sanitize_tradeoffs_and_decisions(raw.get("tradeoffs_and_decisions") or raw.get("key_decisions", []))

    # Side discussions
    side_disc = raw.get("side_discussions")
    side_discussions = str(side_disc).strip() if side_disc and str(side_disc).strip() else None

    # Closing synthesis
    closing_synthesis = str(raw.get("closing_synthesis", "")).strip()

    # Topics discussed for legacy compatibility
    topics = sanitize_string_list(raw.get("topics_discussed", []))
    if not topics and thematic_sections:
        topics = [s.title for s in thematic_sections if s.title]

    # If thematic_sections was empty but topics exist, create basic thematic sections
    if not thematic_sections and topics:
        for t in topics:
            thematic_sections.append(ThematicSection(title=t, content=f"Discussion focused on {t}."))

    return MeetingExtraction(
        title=raw_title,
        meeting_focus=focus,
        summary=focus,
        action_items=actions,
        thematic_sections=thematic_sections,
        tradeoffs_and_decisions=decisions,
        side_discussions=side_discussions,
        closing_synthesis=closing_synthesis,
        key_decisions=decisions,
        topics_discussed=topics,
    )
