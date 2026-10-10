import json
from pathlib import Path
from typing import List, Dict
try:
    from processing.cleaner import clean_segments
except ModuleNotFoundError:
    from cleaner import clean_segments


def create_chunks(
    transcript_data: Dict,
    window_sec: float = 75.0,
    overlap_sec: float = 15.0,
) -> List[Dict]:
    """
    Splits transcript segments into time-based sliding window chunks with overlap.

    Args:
        transcript_data: Dict with 'video_id', 'title', and 'segments'
        window_sec: Target duration in seconds for each chunk (default 75s)
        overlap_sec: Target overlap in seconds between consecutive chunks (default 15s)

    Returns:
        List of chunk dicts:
        {
            "chunk_index": int,
            "video_id": str,
            "title": str,
            "text": str,
            "start": float,
            "end": float,
            "duration": float
        }
    """
    video_id = transcript_data.get("video_id", "unknown")
    title = transcript_data.get("title", "")
    raw_segments = transcript_data.get("segments", [])

    # Step 1: Clean segments (strip [Music], HTML unescape, deduplicate)
    segments = clean_segments(raw_segments)
    if not segments:
        return []

    chunks = []
    chunk_index = 0
    start_idx = 0
    total_segments = len(segments)

    while start_idx < total_segments:
        chunk_snippets = []
        chunk_start_time = segments[start_idx]["start"]
        curr_idx = start_idx

        # Accumulate snippets until we span at least window_sec or run out of snippets
        while curr_idx < total_segments:
            snippet = segments[curr_idx]
            chunk_snippets.append(snippet)

            # Check if this snippet pushes duration to or past target window_sec
            if (snippet["end"] - chunk_start_time) >= window_sec:
                break
            curr_idx += 1

        if not chunk_snippets:
            break

        # Combine text and record exact start and end timestamps
        chunk_text = " ".join(s["text"] for s in chunk_snippets).strip()
        actual_start = chunk_snippets[0]["start"]
        actual_end = chunk_snippets[-1]["end"]
        duration = round(actual_end - actual_start, 2)

        chunks.append({
            "chunk_index": chunk_index,
            "video_id": video_id,
            "title": title,
            "text": chunk_text,
            "start": round(actual_start, 2),
            "end": round(actual_end, 2),
            "duration": duration,
        })
        chunk_index += 1

        # If we reached the end of the video, we're done
        if curr_idx >= total_segments - 1:
            break

        # Step 2: Determine where the NEXT chunk should start (target = actual_end - overlap_sec)
        target_next_start = actual_end - overlap_sec

        # Advance start_idx to the first snippet starting at or after target_next_start
        # Guarantee forward progress by starting at least start_idx + 1
        next_idx = start_idx + 1
        while next_idx < total_segments and segments[next_idx]["start"] < target_next_start:
            next_idx += 1

        start_idx = next_idx

    return chunks


if __name__ == "__main__":
    # Test on our cached 3Blue1Brown video
    cache_path = Path("data/transcripts/aircAruvnKk.json")
    if not cache_path.exists():
        print(f"Cache file {cache_path} not found. Run fetcher.py first.")
    else:
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = create_chunks(data, window_sec=75.0, overlap_sec=15.0)

        print(f"🎬 Video: {data.get('title')}")
        print(f"📊 Created {len(chunks)} chunks from {len(data['segments'])} raw segments\n")

        # Inspect Chunk 0 and Chunk 1 to verify overlap
        print("=" * 60)
        print(f"CHUNK 0 (start: {chunks[0]['start']}s, end: {chunks[0]['end']}s, duration: {chunks[0]['duration']}s):")
        print(chunks[0]["text"][:200] + "...")

        print("=" * 60)
        print(f"CHUNK 1 (start: {chunks[1]['start']}s, end: {chunks[1]['end']}s, duration: {chunks[1]['duration']}s):")
        print(chunks[1]["text"][:200] + "...")

        print("=" * 60)
        overlap = round(chunks[0]["end"] - chunks[1]["start"], 2)
        print(f"⏱️ Overlap between Chunk 0 and Chunk 1: {overlap} seconds (Expected ~15s)")
