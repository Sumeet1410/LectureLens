import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from typing import List, Dict, Optional, Tuple
from dotenv import load_dotenv
from google import genai

from store.vector_store import VectorStore

# Load environment variables (such as GEMINI_API_KEY) from .env
load_dotenv()

DEFAULT_MODEL = "gemini-3.7-flash"
FALLBACK_MODELS = ["gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.8-flash"]
DEFAULT_SIMILARITY_THRESHOLD = 0.50
DEFAULT_TOP_K = 5

SYSTEM_PROMPT = """You are LectureLens, a precise and helpful AI teaching assistant for video lectures.

Your task is to answer the student's question based ONLY on the provided video lecture excerpts.

STRICT RULES:
1. Grounding: Answer ONLY using the facts stated in the excerpts below. Do NOT use outside knowledge or speculate.
2. Not Found: If the excerpts do not contain enough information to answer the question, reply with:
   "I couldn't find this mentioned in the video."
3. Timestamp Citations: Every explanation or fact you state MUST cite the timestamp where it occurs. Format your citations as markdown links with the time code, for example: [02:16](https://youtu.be/VIDEO_ID?t=136).
4. Tone: Clear, educational, and concise. Explain the concepts in an easy-to-understand manner.
"""


def format_timestamp(seconds: float) -> str:
    """Converts seconds into human-readable MM:SS or HH:MM:SS format."""
    total_seconds = int(seconds)
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def make_youtube_url(video_id: str, seconds: float) -> str:
    """Creates a deep-link to the exact second in the YouTube video."""
    return f"https://youtu.be/{video_id}?t={int(seconds)}"


class RAGPipeline:
    def __init__(
        self,
        vector_store: Optional[VectorStore] = None,
        model_name: str = DEFAULT_MODEL,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ):
        """
        Initializes the LectureLens RAG Pipeline.
        
        Args:
            vector_store: Instance of VectorStore (created if not provided).
            model_name: Gemini model name (default: gemini-3.8-flash).
            similarity_threshold: Minimum cosine similarity required to answer.
            top_k: Number of relevant chunks to retrieve.
        """
        self.vector_store = vector_store or VectorStore()
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY is missing. Please set it in your .env file or environment."
            )
        self.llm_client = genai.Client(api_key=api_key)

    def _build_context_prompt(self, chunks: List[Dict], question: str) -> str:
        """Formats retrieved chunks with timestamps into the LLM context prompt."""
        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            start_str = format_timestamp(chunk["start"])
            end_str = format_timestamp(chunk["end"])
            url = make_youtube_url(chunk["video_id"], chunk["start"])
            
            context_parts.append(
                f"--- [Excerpt {i}] ({start_str} - {end_str}) ---\n"
                f"Source Link: {url}\n"
                f"Lecture Title: {chunk.get('title', 'Unknown')}\n"
                f"Transcript Text:\n{chunk['text']}\n"
            )

        context_text = "\n".join(context_parts)
        prompt = (
            f"Here are the relevant lecture excerpts:\n\n"
            f"{context_text}\n"
            f"STUDENT QUESTION:\n{question}\n\n"
            f"Provide a clear, grounded answer with timestamp citations [MM:SS](url):"
        )
        return prompt

    def answer(
        self,
        question: str,
        video_id: Optional[str] = None
    ) -> Dict:
        """
        Executes the full RAG pipeline:
        1. Retrieves top_k chunks from ChromaDB.
        2. Applies similarity guardrail threshold.
        3. Prompts the LLM with strict grounding.
        4. Returns answer and structured citations.
        """
        # 1. Retrieve top chunks
        hits = self.vector_store.search(
            query=question,
            video_id=video_id,
            top_k=self.top_k
        )

        # 2. Similarity threshold guardrail
        # If no hits or top similarity is below threshold, bail out immediately
        best_similarity = hits[0]["similarity"] if hits else 0.0
        if not hits or best_similarity < self.similarity_threshold:
            return {
                "question": question,
                "answer": "I couldn't find this mentioned in the video.",
                "found": False,
                "best_similarity": best_similarity,
                "citations": []
            }

        # 3. Build context and call Gemini with fallback protection
        user_prompt = self._build_context_prompt(hits, question)
        
        answer_text = None
        candidate_models = [self.model_name] + [m for m in FALLBACK_MODELS if m != self.model_name]

        for candidate in candidate_models:
            try:
                response = self.llm_client.models.generate_content(
                    model=candidate,
                    contents=user_prompt,
                    config={
                        "system_instruction": SYSTEM_PROMPT,
                        "temperature": 0.2,
                    }
                )
                if response and response.text:
                    answer_text = response.text.strip()
                    break
            except Exception as e:
                print(f"[WARN] Generation failed with {candidate}: {e}. Trying fallback...")
                continue

        if not answer_text:
            answer_text = "⚠️ The AI service is currently experiencing high demand. Please try asking again in a few moments."

        # 4. Prepare structured citations for UI use
        citations = []
        for hit in hits:
            citations.append({
                "start": hit["start"],
                "end": hit["end"],
                "start_formatted": format_timestamp(hit["start"]),
                "end_formatted": format_timestamp(hit["end"]),
                "url": make_youtube_url(hit["video_id"], hit["start"]),
                "similarity": hit["similarity"],
                "snippet": hit["text"][:200] + "...",
                "chunk_index": hit["chunk_index"],
                "video_id": hit["video_id"]
            })

        return {
            "question": question,
            "answer": answer_text,
            "found": True,
            "best_similarity": best_similarity,
            "citations": citations
        }


if __name__ == "__main__":
    import sys
    # Fix potential Windows terminal UTF-8 encoding issues
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")

    pipeline = RAGPipeline()

    # Test 1: In-domain question that exists in the video
    q1 = "What is a neuron and how does it hold a number?"
    print(f"\n==========================================")
    print(f"❓ Question: {q1}")
    print(f"==========================================")
    res1 = pipeline.answer(q1, video_id="aircAruvnKk")
    print(f"\n🤖 Answer:\n{res1['answer']}")
    print(f"\n📊 Best similarity: {res1['best_similarity']}")
    print(f"📌 Found: {res1['found']}")

    # Test 2: Out-of-domain question that does NOT exist in the video
    q2 = "How do you bake sourdough bread?"
    print(f"\n==========================================")
    print(f"❓ Question: {q2}")
    print(f"==========================================")
    res2 = pipeline.answer(q2, video_id="aircAruvnKk")
    print(f"\n🤖 Answer:\n{res2['answer']}")
    print(f"\n📊 Best similarity: {res2['best_similarity']}")
    print(f"📌 Found: {res2['found']}")
