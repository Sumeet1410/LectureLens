# 🎓 LectureLens — Grounded AI Video Lecture Assistant

> **A RAG-powered study companion that answers student questions about video lectures with exact, clickable timestamp citations.**

---

## 📌 Overview

LectureLens allows students to paste any YouTube lecture URL and ask natural language questions. Instead of generic or hallucinated answers, the system retrieves the most relevant lecture segments, passes them to a strictly grounded LLM, and provides answers cited with working timestamp links (e.g., `[02:16](https://youtu.be/...?t=136)`).

---

## 🚀 Key Features

* **Grounded RAG Pipeline:** Answers are generated **only** from the spoken content of the video lecture.
* **Clickable Timestamp Links:** Jump directly to the exact second in the lecture where the concept is explained.
* **Smart Guardrails:** Automatically blocks out-of-domain questions (e.g. *"How do I bake bread?"*) when similarity is below the threshold, saving API tokens and preventing hallucinations.
* **On-the-Fly Video Ingestion:** Paste any YouTube lecture URL to fetch captions, chunk, and embed them in seconds.
* **Resilient LLM Fallback:** Built-in multi-model fallback chain (`gemini-3.7-flash` → `gemini-3.5-flash` → `gemini-3.8-flash`) to ensure continuous uptime during high-traffic spikes.
* **Interactive Web Interface:** Streamlit UI featuring an embedded lecture player, chat workspace, and expandable source excerpt cards.

---

## 🏗️ Architecture & Pipeline

```
YouTube Video URL
       │
       ▼
1. INGESTION (ingestion/fetcher.py)
   ├── Extracts video ID
   ├── Fetches captions (YouTubeTranscriptApi)
   └── Caches raw transcripts to data/transcripts/
       │
       ▼
2. PROCESSING (processing/chunker.py)
   ├── Cleans text & removes redundant tags
   └── Groups into 60-second sliding windows with 15-second overlap
       │
       ▼
3. VECTOR STORE (store/vector_store.py)
   ├── Embeds chunks using BAAI/bge-small-en-v1.5 (local CPU)
   └── Stores vectors + timestamps in ChromaDB (cosine similarity)
       │
       ▼
4. RETRIEVAL & ANSWER (rag/pipeline.py)
   ├── Embeds user query with BGE search prefix
   ├── Searches Top-5 chunks from ChromaDB
   ├── Similarity threshold guardrail (blocks < 0.50 score)
   └── Strictly prompts Gemini with timestamps & excerpts
       │
       ▼
5. USER INTERFACE (app.py)
   └── Interactive Streamlit app with video player & chat workspace
```

---

## 🛠️ Tech Stack

* **Language:** Python 3.12+
* **Vector Database:** ChromaDB
* **Embedding Model:** `BAAI/bge-small-en-v1.5` (via `sentence-transformers`)
* **LLM Engine:** Google Gemini (`gemini-3.7-flash`) via `google-genai` SDK
* **Frontend:** Streamlit
* **Transcript Source:** `youtube-transcript-api`

---

## ⚡ Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/your-username/LectureLens.git
cd LectureLens
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Key
Create a `.env` file in the root directory:
```env
GEMINI_API_KEY=your_google_ai_studio_api_key_here
```
*(Get a free key from [Google AI Studio](https://aistudio.google.com/)).*

### 5. Launch the Web App
```bash
streamlit run app.py
```
Open **`http://localhost:8501`** in your browser.

---

## ⚠️ Known Limitations

* **Caption Dependency:** Relies on YouTube closed captions existing. Videos without captions cannot be ingested.
* **Auto-Caption Accuracy:** Can occasionally mishear domain-specific jargon or heavy accents.
* **Regional Languages:** Multilingual and code-mixed (e.g., Hinglish) lectures may have lower retrieval accuracy.
* **Cloud IP Restrictions:** YouTube occasionally rate-limits or blocks caption requests from certain cloud server hosting ranges (e.g. AWS/Render). For production cloud demos, pre-cached video transcripts in `data/transcripts/` are recommended.

---

## 🔮 Future Roadmap

1. **Evaluation Baseline:** Automated hit-rate evaluation benchmark across curated lecture questions.
2. **Whisper Fallback:** Ingest raw audio via yt-dlp and transcribe using Whisper API when captions are absent.
3. **Qdrant Migration:** Scale from embedded SQLite ChromaDB to standalone Qdrant vector database.
4. **Streaming Answers:** Token-by-token streaming responses for near-instant perceived latency.
