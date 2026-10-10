import json
import re
import urllib.request
from pathlib import Path
from urllib.parse import parse_qs, urlparse
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
)

# Directory where fetched transcripts will be stored
CACHE_DIR = Path("data/transcripts")


def extract_video_id(url_or_id: str) -> str:
    """
    Extracts the 11-character YouTube video ID from various URL formats or raw ID.

    Supported formats:
    - https://www.youtube.com/watch?v=aircAruvnKk
    - https://youtu.be/aircAruvnKk
    - https://www.youtube.com/embed/aircAruvnKk
    - aircAruvnKk (direct 11-character ID)
    """
    url_or_id = url_or_id.strip()

    # Case 1: Direct 11-character ID passed
    if len(url_or_id) == 11 and re.match(r"^[A-Za-z0-9_-]{11}$", url_or_id):
        return url_or_id

    parsed = urlparse(url_or_id)

    # Case 2: Shortened URL (https://youtu.be/<video_id>)
    if parsed.hostname in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/").split("?")[0]

    # Case 3: Standard YouTube URL (https://www.youtube.com/watch?v=<video_id>)
    if parsed.hostname in ("youtube.com", "www.youtube.com", "m.youtube.com"):
        if parsed.path == "/watch":
            query_params = parse_qs(parsed.query)
            if "v" in query_params:
                return query_params["v"][0]
        # Case 4: Embed URL (https://www.youtube.com/embed/<video_id>)
        elif parsed.path.startswith(("/embed/", "/v/")):
            return parsed.path.split("/")[2]

    raise ValueError(f"Could not extract a valid YouTube video ID from: {url_or_id}")


def get_video_title(video_id: str) -> str:
    """
    Fetches the public title of the video using YouTube's free oEmbed endpoint.
    Requires no API keys. Falls back to a default name if offline/blocked.
    """
    oembed_url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    req = urllib.request.Request(oembed_url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(req, timeout=5) as response:
            data = json.loads(response.read().decode("utf-8"))
            return data.get("title", f"Video {video_id}")
    except Exception:
        return f"Video {video_id}"


def get_transcript(url_or_id: str) -> dict:
    """
    Fetches the transcript and title for a YouTube video.
    
    1. Checks the local cache (data/transcripts/{video_id}.json).
    2. If not found, fetches title via oEmbed and captions via YouTubeTranscriptApi.
       Prefers human-made captions before falling back to auto-generated ones.
    3. Normalizes each snippet into Common Format: {text, start, end}.
    4. Caches the result to disk as JSON for instant future reuse.
    
    Returns:
        dict: {
            "video_id": str,
            "title": str,
            "segments": list of {text, start, end}
        }
    """
    video_id = extract_video_id(url_or_id)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE_DIR / f"{video_id}.json"

    # 1. Cache hit: Read directly from local disk
    if cache_file.exists():
        print(f"[CACHE] Loading transcript from {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            cached_data = json.load(f)
            # Handle legacy cache format (list) if previously saved
            if isinstance(cached_data, list):
                title = get_video_title(video_id)
                cached_data = {
                    "video_id": video_id,
                    "title": title,
                    "segments": cached_data
                }
                with open(cache_file, "w", encoding="utf-8") as out:
                    json.dump(cached_data, out, indent=2, ensure_ascii=False)
            return cached_data

    # 2. Cache miss: Fetch title and captions
    print(f"[FETCH] Downloading captions for ID: {video_id}...")
    title = get_video_title(video_id)

    api = YouTubeTranscriptApi()
    try:
        transcript_list = api.list(video_id)
        # Priority 3: Try to find manual captions first, then fallback to auto-generated
        try:
            transcript_obj = transcript_list.find_manually_created_transcript(["en", "en-US", "en-GB"])
        except NoTranscriptFound:
            transcript_obj = transcript_list.find_generated_transcript(["en", "en-US", "en-GB"])

        fetched_snippets = transcript_obj.fetch()
    except TranscriptsDisabled:
        raise RuntimeError(f"Captions are disabled by the creator for video '{title}' ({video_id}).")
    except NoTranscriptFound:
        raise RuntimeError(f"No English transcripts (manual or auto-generated) found for video '{title}' ({video_id}).")
    except VideoUnavailable:
        raise RuntimeError(f"The video ({video_id}) is unavailable or private.")
    except Exception as e:
        raise RuntimeError(f"Failed to fetch transcript for ({video_id}): {e}")

    # 3. Standardize to Common Format: {text, start, end}
    standardized_segments = []
    for snippet in fetched_snippets:
        # Handles snippet whether it's dict or FetchedTranscriptSnippet object
        text = snippet["text"].strip() if isinstance(snippet, dict) else snippet.text.strip()
        start = snippet["start"] if isinstance(snippet, dict) else snippet.start
        duration = snippet["duration"] if isinstance(snippet, dict) else snippet.duration

        if text:
            start_time = round(start, 2)
            end_time = round(start + duration, 2)
            standardized_segments.append({
                "text": text,
                "start": start_time,
                "end": end_time
            })

    result = {
        "video_id": video_id,
        "title": title,
        "segments": standardized_segments
    }

    # 4. Save to cache
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved '{title}' ({len(standardized_segments)} segments) to {cache_file}")

    return result


if __name__ == "__main__":
    # Test with 3Blue1Brown's Neural Networks intro
    sample_url = "https://www.youtube.com/watch?v=aircAruvnKk"
    data = get_transcript(sample_url)
    print(f"\nTitle: {data['title']}")
    print(f"Video ID: {data['video_id']}")
    print(f"Total segments: {len(data['segments'])}")
    print("\nFirst 3 segments:")
    for seg in data["segments"][:3]:
        print(seg)

