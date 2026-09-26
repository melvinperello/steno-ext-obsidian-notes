"""SRT subtitle cleaning, confidence filtering, and foreign script hallucination removal.
Internal pre-processing utility for obsidian_notes prior to LLM distillation.
"""

from __future__ import annotations

import re
from typing import List, Tuple

# Regex to detect foreign scripts frequently hallucinated by Whisper on noise/silence
FOREIGN_SCRIPT_REGEX = re.compile(
    r"[\u4e00-\u9fff"  # Chinese Unified Ideographs
    r"\u3400-\u4dbf"  # Chinese Extension A
    r"\u3040-\u309f"  # Japanese Hiragana
    r"\u30a0-\u30ff"  # Japanese Katakana
    r"\u0900-\u097f"  # Devanagari / Sanskrit
    r"\ua8e0-\ua8ff"  # Devanagari Extended
    r"\uac00-\ud7af"  # Korean Hangul Syllables
    r"\u1100-\u11ff"  # Korean Hangul Jamo
    r"\u0600-\u06ff"  # Arabic
    r"\u0e00-\u0e7f]"  # Thai
)

# Common Whisper repetitive hallucination loops on silence
COMMON_HALLUCINATION_PATTERNS = [
    re.compile(r"(?:thank you for watching[\.\!\s]*){2,}", re.IGNORECASE),
    re.compile(r"(?:subtitles by\s+.*?){2,}", re.IGNORECASE),
    re.compile(r"(?:please subscribe[\.\!\s]*){2,}", re.IGNORECASE),
    re.compile(r"(?:\[music\][\s]*){3,}", re.IGNORECASE),
]


def is_hallucinated_text(text: str) -> bool:
    """Checks if a subtitle text line consists predominantly of hallucinated foreign scripts or loops."""
    stripped = text.strip()
    if not stripped:
        return False

    # Check for foreign script characters (Chinese, Japanese, Sanskrit, etc.)
    foreign_chars = FOREIGN_SCRIPT_REGEX.findall(stripped)
    if foreign_chars:
        non_space_len = len(re.sub(r"\s+", "", stripped))
        if non_space_len > 0 and (len(foreign_chars) / non_space_len) > 0.15:
            return True

    # Check for common repetitive looping hallucinations
    for pattern in COMMON_HALLUCINATION_PATTERNS:
        if pattern.search(stripped):
            return True

    # Check for excessive repetition of identical short tokens (e.g. "you you you you you")
    words = stripped.split()
    if len(words) >= 6:
        word_counts = {}
        for w in words:
            word_counts[w.lower()] = word_counts.get(w.lower(), 0) + 1
        most_common_count = max(word_counts.values())
        if most_common_count / len(words) > 0.6:
            return True

    return False


def clean_subtitle_line(text: str) -> str:
    """Strips foreign hallucinated characters from an otherwise legitimate line."""
    cleaned = FOREIGN_SCRIPT_REGEX.sub("", text).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def parse_srt_blocks(srt_content: str) -> List[Tuple[str, str, str]]:
    """Parses raw SRT content into (index, timecode, text) blocks."""
    blocks = []
    raw_blocks = re.split(r"\n\s*\n", srt_content.strip())

    for b in raw_blocks:
        lines = [line.strip() for line in b.strip().splitlines() if line.strip()]
        if len(lines) >= 2 and "-->" in lines[1]:
            idx = lines[0]
            timecode = lines[1]
            text = "\n".join(lines[2:])
            blocks.append((idx, timecode, text))
        elif len(lines) >= 1 and "-->" in lines[0]:
            timecode = lines[0]
            text = "\n".join(lines[1:])
            blocks.append(("", timecode, text))

    return blocks


def filter_and_clean_srt(
    srt_content: str,
    strip_foreign_scripts: bool = True,
    strip_loops: bool = True,
) -> str:
    """Filters out Chinese, Japanese, Sanskrit hallucinations and repetitive Whisper noise.

    Returns:
        Cleaned, re-numbered SRT subtitle content for LLM distillation.
    """
    blocks = parse_srt_blocks(srt_content)
    cleaned_blocks = []
    current_idx = 1

    for _, timecode, text in blocks:
        if is_hallucinated_text(text):
            cleaned_blocks.append(f"{current_idx}\n{timecode}\n[UNCLEAR_AUDIO]")
            current_idx += 1
            continue

        cleaned_text = clean_subtitle_line(text)
        if not cleaned_text:
            cleaned_text = "[UNCLEAR_AUDIO]"

        cleaned_blocks.append(f"{current_idx}\n{timecode}\n{cleaned_text}")
        current_idx += 1

    return "\n\n".join(cleaned_blocks) + "\n"
