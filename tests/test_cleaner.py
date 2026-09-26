try:
    from obsidian_notes.cleaner import filter_and_clean_srt, is_hallucinated_text
except ImportError:
    from cleaner import filter_and_clean_srt, is_hallucinated_text


def test_is_hallucinated_chinese_japanese_sanskrit():
    # Chinese hallucination on noise
    assert is_hallucinated_text("你好世界，这是测试。") is True
    # Sanskrit / Devanagari hallucination
    assert is_hallucinated_text("ॐ भूर्भुवः स्वः तत्सवितुर्वरेण्यं") is True
    # Japanese Hiragana / Katakana hallucination
    assert is_hallucinated_text("ありがとうございました") is True
    # Repetitive loop hallucination
    assert is_hallucinated_text("Thank you for watching. Thank you for watching. Thank you for watching.") is True

    # Normal Latin conversation should NOT be flagged
    assert is_hallucinated_text("Let's review the architecture for steno-svc today.") is False
    assert is_hallucinated_text("Kumusta, magandang umaga sa lahat.") is False
    assert is_hallucinated_text("Buenos días, vamos a revisar el proyecto.") is False


def test_filter_and_clean_srt():
    dirty_srt = """1
00:00:00,500 --> 00:00:03,000
Hello everyone, welcome to the meeting.

2
00:00:03,200 --> 00:00:06,000
你好世界这是一个幻觉测试

3
00:00:06,500 --> 00:00:09,000
ॐ नमः शिवाय

4
00:00:09,500 --> 00:00:12,000
We decided to deploy the service on Windows.
"""

    cleaned = filter_and_clean_srt(dirty_srt)

    # Legitimate lines are preserved
    assert "Hello everyone, welcome to the meeting." in cleaned
    assert "We decided to deploy the service on Windows." in cleaned

    # Foreign hallucinations replaced with [UNCLEAR_AUDIO]
    assert "你好世界" not in cleaned
    assert "ॐ नमः शिवाय" not in cleaned
    assert "[UNCLEAR_AUDIO]" in cleaned
