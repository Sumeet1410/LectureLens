# Lecture Video Study Assistant: Project Plan

**Goal:** paste a YouTube lecture link, ask questions, get answers grounded in the transcript with clickable timestamp links.

**Approach:** build the simplest working version (Plan A, the Base Project), measure it with an eval set, then add upgrades one at a time in order of increasing complexity. Every upgrade must be justified by a number.

---

## Plan A: Base Project

### What it does
1. User pastes a YouTube link
2. App fetches captions, cleans them, chunks, embeds, and stores them
3. User asks questions
4. App retrieves the top chunks and an LLM answers only from them
5. Answer includes timestamp links (`https://youtu.be/VIDEO_ID?t=754`) and the supporting chunk text

### Decisions (beginner defaults)

| Stage | Choice |
|---|---|
| Transcript source | `youtube-transcript-api` (captions only) |
| Common format | `{text, start, end}` per segment, where `end = start + duration` |
| Cleaning | Dedupe overlapping lines, strip `[Music]` style tags, normalize whitespace. Keep raw and cleaned versions |
| Chunking | Merge snippets into ~75s windows, ~15s overlap, no sentence logic |
| Chunk metadata | `text`, `start`, `end`, `video_id`, `title`, `chunk_index` |
| Embeddings | `bge-small-en-v1.5` (local). Remember the query prefix from the model card |
| Vector DB | ChromaDB |
| Retrieval | Top-5 cosine similarity |
| Not found | Similarity threshold; below it, say "I couldn't find this in the video" |
| LLM | Any API you have; prompt it to answer only from the provided chunks |
| UI | Streamlit or Gradio |
| Caching | Save transcript JSON to `data/transcripts/{video_id}.json` and skip ingestion if it exists |

### Build order

**Phase 1: Transcript fetcher.** `get_transcript(url)`: extract video ID, check cache, fetch captions, convert to the common format, save JSON. Print one result to see the real object.

**Phase 2: Cleaning and chunking.** Dedupe, strip tags, then merge snippets into windows with overlap. Test that chunk `start` is the first snippet's start and `end` is the last snippet's end.

**Phase 3: Embed and store.** Embed each chunk, store with metadata in ChromaDB.

**Phase 4: Retrieve and answer.** Embed the question, fetch top-5, build the prompt, return the answer with timestamp links.

**Phase 5: UI.** Link box, progress messages during ingestion, chat box, clickable timestamps.

**Phase 6: Eval set and baseline.** Write 20-30 questions across 2-3 lectures you know well, with correct timestamp ranges (include a few not-in-video questions). Write a script that computes top-5 hit rate: a hit means any retrieved chunk overlaps the correct range. Record your baseline number. **Ideally build the eval set before tuning anything, even earlier than Phase 6.**

### Suggested folder structure

```
lecture-rag/
├── ingestion/      # transcript fetching, caching
├── processing/     # cleaning, chunking
├── store/          # embedding + vector DB
├── rag/            # retrieval, prompts, answers
├── eval/           # test questions + scoring script
├── data/transcripts/
└── app.py
```

### Definition of done
- Paste a link and get a cited answer with working timestamp links
- Eval set exists and a baseline hit rate is recorded
- README explains the pipeline and known limits

### Known limits (state these honestly in the README)
- Depends on captions existing; videos without them fail with a clear message
- Auto-caption errors on Indian-accented speech and technical terms
- Hinglish and regional languages unreliable
- One video at a time
- Caption fetching may be blocked on cloud servers, so the demo should use pre-ingested videos

---

## Future Scope: Potential Upgrades

Ordered from least to most complex. Change **one variable per run** and re-run the eval set after each. Keep only what helps.

| # | Upgrade | Complexity | What it fixes | Expected impact |
|---|---|---|---|---|
| 1 | **Chunk parameter experiments**: test 45s / 75s / 120s windows and different overlaps; use pause gaps as soft cut points | Very low | Chunks too fuzzy or too fragmented | Medium, and a great README table |
| 2 | **Embedding model comparison**: try `bge-base`, then one other model | Very low | Weak semantic matching | Low-medium |
| 3 | **Neighbor expansion**: also pull the chunks just before and after each hit | Low | Answers cut off at chunk edges | Medium |
| 4 | **Whisper fallback via a hosted API (e.g. Groq)**: yt-dlp downloads audio, Whisper returns timestamped segments, slotted into the fallback chain. Split long audio and offset timestamps | Low-medium | No captions, bad captions, missing punctuation | **High** for Indian-accented lectures |
| 5 | **Whisper term prompt (`initial_prompt`)**: feed domain terms (NumPy, gradient descent) | Low-medium | Mis-heard technical terms | Medium |
| 6 | **Paste-any-link ingestion with progress UI**: `ingest(url)` triggered from the app, with video length cap | Medium | Limited to pre-loaded videos | Product value, not retrieval quality |
| 7 | **Summaries (map-reduce)**: summarize chunks, then summarize the summaries, by section with timestamps | Medium | "Summarize this video" can't use top-k | Feature |
| 8 | **Quiz generation**: LLM returns JSON questions grounded in specific chunks, validated, with a timestamp to review | Medium | Study-mode feature | Feature |
| 9 | **Hybrid search (BM25 + vectors)** | Medium | Exact terms, names, acronyms | Medium-high |
| 10 | **Reranker (cross-encoder)**: retrieve ~20, rescore, keep best 5 | Medium | Right chunk retrieved but ranked low | Medium-high |
| 11 | **MMR**: reduce near-duplicate overlapping results | Medium | Five chunks saying the same thing | Low-medium |
| 12 | **LLM transcript cleanup** (using video title as context; keep raw too) | Medium | Remaining term errors | Medium, but can introduce its own errors |
| 13 | **Conversation memory and query rewriting** | Medium-high | Follow-ups like "what about the second one?" | Feature |
| 14 | **Punctuation restoration + sentence-aware chunking** | Medium-high | Chunks cutting mid-thought | Low if using Whisper already |
| 15 | **Intent routing**: detect answer vs summary vs quiz | Medium-high | Manual mode switching | Feature |
| 16 | **Parent-child retrieval**: search small chunks, give the LLM the larger surrounding chunk | High | Precision vs context trade-off | Medium-high |
| 17 | **Playlist / multi-video support with metadata filtering** (by `video_id`, time range) | High | One-video limit | Feature |
| 18 | **Topic-shift (semantic) chunking**: embed sentences, split where neighbor similarity drops | High | Chunks spanning multiple topics | Medium |
| 19 | **Hinglish / Indian-language support**: `bge-m3` embeddings, Sarvam AI or similar speech APIs, multilingual eval questions | High | Code-mixed lectures | High for that audience |
| 20 | **Slide OCR**: extract on-screen text from video frames and index it with timestamps | Very high | Content shown but not spoken | Niche, impressive |
| 21 | **Cloud deployment with real ingestion**: handle IP blocking, background jobs, cost limits | Very high | Public demo reliability | Product value |

### Suggested stopping points
- **Minimum portfolio version:** Plan A + upgrades 1, 3, 4, and the benchmark table
- **Strong version:** add 6, 7, 8, 9, 10
- **Stretch:** 13 onward, picked by interest

---

## Benchmark table to put in your README

| Configuration | Top-5 hit rate |
|---|---|
| Plan A baseline (captions, 75s, bge-small) | X% |
| + best chunk size | X% |
| + Whisper | X% |
| + Whisper with term prompt | X% |
| + hybrid search | X% |
| + reranker | X% |

The story to tell: *"I measured how transcript quality and retrieval choices affected accuracy on Indian-accented lectures, and improved hit rate from X to Z."*

---

## Rules to keep you sane
1. Always keep the raw transcript alongside the cleaned one.
2. Use the same embedding model for chunks and questions; changing it means re-embedding everything.
3. Never edit the eval set after seeing results.
4. Keep each stage behind its own function so any piece can be swapped.
5. If copied code fails, check library versions first (`youtube-transcript-api` changed its API between 0.x and 1.x).
6. Downloading content has terms-of-service implications, so keep it to personal and learning use.
