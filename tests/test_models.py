try:
    from obsidian_notes.models import (
        ActionItem,
        MeetingExtraction,
        clean_action_item_fields,
        sanitize_action_items,
        sanitize_meeting_payload,
        sanitize_tradeoffs_and_decisions,
    )
except ImportError:
    from models import (
        ActionItem,
        MeetingExtraction,
        clean_action_item_fields,
        sanitize_action_items,
        sanitize_meeting_payload,
        sanitize_tradeoffs_and_decisions,
    )


def test_action_item_dataclass():
    item = ActionItem(assignee="DevOps", task="Configure NSSM service", deadline="Friday")
    assert item.assignee == "DevOps"
    assert item.task == "Configure NSSM service"
    assert item.deadline == "Friday"

    # Test defaults
    item_default = ActionItem(task="Simple task")
    assert item_default.assignee == "Unassigned"
    assert item_default.deadline == "None specified"
    assert item_default.to_dict()["task"] == "Simple task"


def test_meeting_extraction_dataclass():
    extraction = MeetingExtraction(
        title="Project Kickoff Discussion",
        summary="Briefing on daemon architecture and milestones.",
        key_decisions=["Use smolagents for note distillation", "Implement Syncthing web UI"],
        action_items=[ActionItem(task="Deploy service")],
        topics_discussed=["Architecture", "Deployment"],
    )

    data = extraction.to_dict()
    assert data["title"] == "Project Kickoff Discussion"
    assert len(data["key_decisions"]) == 2
    assert len(data["action_items"]) == 1


def test_sanitize_meeting_payload():
    messy_llm_json = {
        "title": "Messy Voice Memo",
        "summary": "Everything went fine.",
        "key_decisions": "Single decision as string",
        "action_items": [
            {"task": "Buy coffee", "assignee": "DevOps"},
            "Task as plain string without dict",
        ],
        "topics_discussed": ["Topic 1", "Topic 2"],
    }

    sanitized = sanitize_meeting_payload(messy_llm_json, "recording.m4a")
    assert sanitized.title == "Messy Voice Memo"
    assert sanitized.summary == "Everything went fine."
    assert sanitized.key_decisions == ["Single decision as string"]
    assert len(sanitized.action_items) == 2
    assert sanitized.action_items[0].task == "Buy coffee"
    assert sanitized.action_items[0].assignee == "DevOps"
    assert sanitized.action_items[1].task == "Task as plain string without dict"


def test_sanitize_tradeoffs_and_decisions():
    dict_items = [
        {
            "decision_or_tradeoff": "Support queue routing versus core engineering action",
            "outcome": "No blanket engineering commitment was made.",
        },
        {
            "decision": "Use Central Admin Portal",
            "tradeoff": "Requires team training",
        },
        "Plain string decision",
        "{'decision': 'Stringified dict', 'rationale': 'Because reasons'}",
    ]

    res = sanitize_tradeoffs_and_decisions(dict_items)
    assert len(res) == 4
    assert res[0] == "**Support queue routing versus core engineering action**: No blanket engineering commitment was made."
    assert res[1] == "**Use Central Admin Portal** — *Trade-off*: Requires team training"
    assert res[2] == "Plain string decision"
    assert res[3] == "**Stringified dict**: Because reasons"


def test_clean_action_item_fields():
    # 1. Unidentified speaker prefix with (Due: Not stated)
    item1 = clean_action_item_fields(
        task="Unidentified speaker: Inform the team that the referenced parameter can be configured. (Due: Not stated)",
        assignee="Unidentified speaker",
        deadline="Not stated",
    )
    assert item1.task == "Inform the team that the referenced parameter can be configured."
    assert item1.assignee == "Unassigned"
    assert item1.deadline == "None specified"

    # 2. Named assignee with unidentified follow-up
    item2 = clean_action_item_fields(
        task="Lead Architect (assignment); unidentified speaker (follow-up): Assign the referenced item to the speaker. (Due: Not stated)",
        assignee="Unassigned",
        deadline="Not stated",
    )
    assert item2.task == "Assign the referenced item to the speaker."
    assert item2.assignee == "Lead Architect"
    assert item2.deadline == "None specified"

    # 3. Legitimate assignee and deadline preserved
    item3 = clean_action_item_fields(
        task="Review the deployment pipeline",
        assignee="Platform Engineer",
        deadline="Sprint 42",
    )
    assert item3.task == "Review the deployment pipeline"
    assert item3.assignee == "Platform Engineer"
    assert item3.deadline == "Sprint 42"

    # 4. Batch sanitization test
    raw_items = [
        {"task": "Unidentified ticket/requester liaison: Follow up with requesters. (Due: Not stated)"},
        "Release Manager: Deploy service to staging",
        {"task": "Run tests", "assignee": "AWS", "deadline": "Friday"},
    ]
    cleaned = sanitize_action_items(raw_items)
    assert len(cleaned) == 3
    assert cleaned[0].task == "Follow up with requesters."
    assert cleaned[0].assignee == "Unassigned"
    assert cleaned[1].task == "Deploy service to staging"
    assert cleaned[1].assignee == "Release Manager"
    assert cleaned[2].task == "Run tests"
    assert cleaned[2].assignee == "AWS"
    assert cleaned[2].deadline == "Friday"
