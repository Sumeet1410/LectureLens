import os
import sys
from pathlib import Path

# Safeguard Windows console encoding
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import streamlit as st

# Ensure root directory is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ingestion.fetcher import get_transcript, extract_video_id
from processing.chunker import create_chunks
from store.vector_store import VectorStore
from rag.pipeline import RAGPipeline, format_timestamp

# Sample video for 1-click testing
SAMPLE_URL = "https://www.youtube.com/watch?v=aircAruvnKk"

st.set_page_config(
    page_title="LectureLens | AI Lecture Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #888888;
        margin-bottom: 1.5rem;
    }
    .citation-badge {
        display: inline-block;
        padding: 4px 10px;
        background-color: #2b313e;
        border-radius: 6px;
        font-size: 0.85rem;
        margin-right: 6px;
        margin-top: 4px;
    }
    .video-card {
        padding: 12px;
        border-radius: 8px;
        background-color: #1a1c24;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_rag_engine():
    """Initializes and caches the VectorStore and RAG Pipeline in memory."""
    store = VectorStore()
    pipeline = RAGPipeline(vector_store=store)
    return store, pipeline


def process_and_ingest(url: str, store: VectorStore) -> dict:
    """Runs full pipeline: Fetch -> Chunk -> Store."""
    video_id = extract_video_id(url)

    # 1. Fetch transcript
    transcript_data = get_transcript(url)
    
    # 2. Chunk transcript
    chunks = create_chunks(transcript_data)
    
    # 3. Store into ChromaDB
    store.store_chunks(chunks)
    
    return {
        "video_id": video_id,
        "title": transcript_data.get("title", f"Video {video_id}"),
        "num_chunks": len(chunks),
        "url": url
    }


def main():
    store, pipeline = load_rag_engine()

    # Initialize session states
    if "current_video" not in st.session_state:
        # Default to pre-indexed 3Blue1Brown video
        st.session_state.current_video = {
            "video_id": "aircAruvnKk",
            "title": "What is a neural network? | Chapter 1, Deep learning",
            "url": SAMPLE_URL
        }

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- SIDEBAR: Ingestion & Video Selection ---
    with st.sidebar:
        st.header("📺 Video Management")
        st.markdown("Paste any YouTube lecture to index and chat with:")

        input_url = st.text_input(
            "YouTube URL",
            value="",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Supports standard and youtu.be links"
        )

        col1, col2 = st.columns([1, 1])
        with col1:
            load_btn = st.button("🚀 Ingest Video", use_container_width=True, type="primary")
        with col2:
            sample_btn = st.button("✨ Sample Video", use_container_width=True)

        if sample_btn:
            input_url = SAMPLE_URL
            load_btn = True

        if load_btn:
            target_url = input_url.strip() if input_url.strip() else SAMPLE_URL
            with st.status("🔄 Ingesting lecture...", expanded=True) as status:
                try:
                    st.write("📥 Fetching captions & video metadata...")
                    info = process_and_ingest(target_url, store)
                    
                    st.write(f"✂️ Chunked & embedded into ChromaDB ({info['num_chunks']} chunks)...")
                    st.session_state.current_video = info
                    st.session_state.messages = []  # reset chat for new video
                    status.update(label="✅ Video indexed and ready!", state="complete", expanded=False)
                    st.success(f"Ready: **{info['title']}**")
                except Exception as e:
                    status.update(label="❌ Ingestion failed", state="error")
                    st.error(f"Error: {e}")

        st.divider()

        # Active Video Card
        active_vid = st.session_state.current_video
        st.markdown(f"**Current Lecture:**\n\n📌 **{active_vid['title']}**")
        st.caption(f"ID: `{active_vid['video_id']}`")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # --- MAIN VIEW ---
    st.markdown('<div class="main-title">🎓 LectureLens</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Grounded AI teaching assistant with clickable video citations</div>', unsafe_allow_html=True)

    left_col, right_col = st.columns([1, 1.2], gap="large")

    # LEFT COLUMN: Embedded Player
    with left_col:
        st.subheader("🎥 Lecture Player")
        st.video(st.session_state.current_video["url"])
        
        st.markdown("""
        **💡 Suggested Questions to Try:**
        * *What is a neuron and what does its activation represent?*
        * *Why do we use the sigmoid squishification function?*
        * *What do the weights and biases in the network actually do?*
        * *Who won the 2022 World Cup?* (Tests out-of-domain guardrail)
        """)

    # RIGHT COLUMN: Chat Workspace
    with right_col:
        st.subheader("💬 Ask Questions")

        # Container for chat messages
        chat_container = st.container(height=480)
        with chat_container:
            if not st.session_state.messages:
                st.info("👋 Ask anything about this lecture! Answers will cite exact timestamps.")

            for msg in st.session_state.messages:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])
                    
                    # Display structured citations if available
                    if msg.get("citations"):
                        st.markdown("🎯 **Timestamps in video:**")
                        badge_links = [
                            f"[{c['start_formatted']} - {c['end_formatted']}]({c['url']})"
                            for c in msg["citations"]
                        ]
                        st.markdown(" • ".join(badge_links))

                        # Expandable source chunks
                        with st.expander(f"📚 View {len(msg['citations'])} Retrieved Excerpts"):
                            for i, c in enumerate(msg["citations"], 1):
                                st.markdown(f"**Excerpt {i}** ({c['start_formatted']} - {c['end_formatted']}) — *Match: {int(c['similarity']*100)}%*")
                                st.caption(c["snippet"])
                                st.divider()

        # Chat Input
        if user_query := st.chat_input("Ask a question about the video..."):
            # Display user message
            st.session_state.messages.append({"role": "user", "content": user_query})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Searching transcript and formulating answer..."):
                        try:
                            res = pipeline.answer(
                                question=user_query,
                                video_id=st.session_state.current_video["video_id"]
                            )
                        except Exception as e:
                            res = {
                                "answer": f"⚠️ Could not generate an answer due to temporary service issue: {e}",
                                "citations": []
                            }

                        st.markdown(res["answer"])

                        if res.get("citations"):
                            st.markdown("🎯 **Timestamps in video:**")
                            badge_links = [
                                f"[{c['start_formatted']} - {c['end_formatted']}]({c['url']})"
                                for c in res["citations"]
                            ]
                            st.markdown(" • ".join(badge_links))

                            with st.expander(f"📚 View {len(res['citations'])} Retrieved Excerpts"):
                                for i, c in enumerate(res["citations"], 1):
                                    st.markdown(f"**Excerpt {i}** ({c['start_formatted']} - {c['end_formatted']}) — *Match: {int(c['similarity']*100)}%*")
                                    st.caption(c["snippet"])
                                    st.divider()

            # Save assistant response to state
            st.session_state.messages.append({
                "role": "assistant",
                "content": res["answer"],
                "citations": res.get("citations", [])
            })


if __name__ == "__main__":
    main()
