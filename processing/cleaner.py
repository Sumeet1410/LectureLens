import html
import re
from typing import List, Dict


def clean_text(text: str) -> str:
    """
    Cleans an individual text snippet:
    1. Unescapes HTML entities (e.g., &amp; -> &, &#39; -> ')
    2. Strips audio/visual cue tags like [Music], [Applause], (cheers)
    3. Normalizes whitespace (collapses tabs, newlines, and multiple spaces into one)
    """
    if not text:
        return ""

    # 1. Unescape HTML artifacts
    cleaned = html.unescape(text)

    # 2. Remove bracketed sound tags like [Music], [Applause], [laughter]
    cleaned = re.sub(r"\[.*?\]", "", cleaned)

    # Also remove parenthesized sound tags like (music), (applause) if lowercase
    cleaned = re.sub(r"\((music|applause|laughter|cheering|inaudible)\)", "", cleaned, flags=re.IGNORECASE)

    # 3. Normalize all whitespace (newlines, tabs, multiple spaces -> single space)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned


def clean_segments(segments: List[Dict]) -> List[Dict]:
    """
    Cleans a list of transcript segments ({text, start, end}):
    - Cleans text using clean_text()
    - Drops segments that become empty after cleaning (e.g. music-only snippets)
    - Deduplicates identical consecutive text lines common in auto-captions
    """
    cleaned_segments = []
    last_text = None

    for seg in segments:
        cleaned_str = clean_text(seg.get("text", ""))

        # Skip empty snippets (e.g. snippets that only contained [Music])
        if not cleaned_str:
            continue

        # Skip consecutive duplicates (auto-generated captions often repeat lines)
        if cleaned_str == last_text:
            continue

        cleaned_segments.append({
            "text": cleaned_str,
            "start": seg["start"],
            "end": seg["end"]
        })
        last_text = cleaned_str

    return cleaned_segments


if __name__ == "__main__":
    # Test cases demonstrating each cleaning feature
    dirty_samples = [
        {"text": "[Music] Welcome everyone! &amp; hello.", "start": 0.0, "end": 2.5},
        {"text": "Welcome everyone! &amp; hello.", "start": 2.5, "end": 5.0},  # duplicate line test
        {"text": "[Applause]", "start": 5.0, "end": 7.0},  # empty drop test
        {"text": "Today we&#39;re talking about   \n\n  neural networks.", "start": 7.0, "end": 10.0},
    ]

    print("--- BEFORE CLEANING ---")
    for s in dirty_samples:
        print(s)

    cleaned = clean_segments(dirty_samples)

    print("\n--- AFTER CLEANING ---")
    for s in cleaned:
        print(s)
