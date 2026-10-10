import os
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Safeguard Windows stdout against encoding crashes with emojis
if sys.platform == "win32":
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        if hasattr(sys.stderr, "reconfigure"):
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# Directory where ChromaDB will persist its database files on disk
CHROMA_PATH = Path("data/chroma_db")
COLLECTION_NAME = "lecture_chunks"

# Model name specified in the project plan
MODEL_NAME = "BAAI/bge-small-en-v1.5"

# BGE models require a special prefix for search queries, but NOT for stored passages
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class VectorStore:
    def __init__(self, persist_dir: Path = CHROMA_PATH):
        """
        Initializes the VectorStore:
        1. Loads the local bge-small-en-v1.5 embedding model into memory.
        2. Connects to ChromaDB's persistent storage on local disk.
        """
        print(f"[MODEL] Loading embedding model ({MODEL_NAME})...")
        self.embedder = SentenceTransformer(MODEL_NAME)

        print(f"[CHROMA] Initializing ChromaDB at {persist_dir}...")
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        
        # Get or create our unified collection
        # Uses cosine similarity as specified in the plan
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"}
        )

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Generates embedding vectors for a list of passage texts (no query prefix)."""
        embeddings = self.embedder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Generates an embedding vector for a search query with the required BGE prefix."""
        prefixed_query = BGE_QUERY_PREFIX + query.strip()
        embedding = self.embedder.encode(prefixed_query, normalize_embeddings=True, show_progress_bar=False)
        return embedding.tolist()

    def store_chunks(self, chunks: List[Dict]) -> int:
        """
        Embeds and stores chunks into ChromaDB.
        Uses upsert so re-running on the same video updates rather than duplicates.
        """
        if not chunks:
            return 0

        ids = []
        documents = []
        metadatas = []

        for chunk in chunks:
            # Unique ID: e.g. "aircAruvnKk_chunk_0"
            chunk_id = f"{chunk['video_id']}_chunk_{chunk['chunk_index']}"
            ids.append(chunk_id)
            documents.append(chunk["text"])

            # Metadata stored alongside the vector for retrieval & UI citations
            metadatas.append({
                "chunk_index": int(chunk["chunk_index"]),
                "video_id": str(chunk["video_id"]),
                "title": str(chunk.get("title", "")),
                "start": float(chunk["start"]),
                "end": float(chunk["end"]),
                "duration": float(chunk.get("duration", round(chunk["end"] - chunk["start"], 2)))
            })

        print(f"[EMBED] Generating embeddings for {len(chunks)} chunks...")
        embeddings = self.embed_texts(documents)

        print(f"[CHROMA] Upserting into collection '{COLLECTION_NAME}'...")
        self.collection.upsert(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas
        )

        print(f"[STORE] Successfully stored {len(chunks)} chunks in vector database.")
        return len(chunks)

    def search(
        self,
        query: str,
        video_id: Optional[str] = None,
        top_k: int = 5
    ) -> List[Dict]:
        """
        Finds the top_k most relevant chunks for a question.
        
        Args:
            query: The user's question
            video_id: Optional filter to search within a specific video
            top_k: Number of chunks to retrieve (default 5, as in Plan A)

        Returns:
            List of result dictionaries containing text, metadata, and similarity score.
        """
        query_vector = self.embed_query(query)

        # Optional metadata filter (e.g. search only within this video)
        where_filter = {"video_id": video_id} if video_id else None

        results = self.collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results = []
        if results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0]
            distances = results["distances"][0]

            for doc, meta, dist in zip(docs, metas, distances):
                # For cosine distance: cosine similarity = 1 - cosine_distance
                similarity = round(1.0 - dist, 4)
                formatted_results.append({
                    "text": doc,
                    "metadata": meta,
                    "similarity": similarity,
                    "start": meta["start"],
                    "end": meta["end"],
                    "title": meta["title"],
                    "video_id": meta["video_id"],
                    "chunk_index": meta["chunk_index"]
                })

        return formatted_results


if __name__ == "__main__":
    # Test storing our processed chunks and performing a query
    import json
    try:
        from processing.chunker import create_chunks
    except ModuleNotFoundError:
        import sys
        sys.path.insert(0, ".")
        from processing.chunker import create_chunks

    cache_file = Path("data/transcripts/aircAruvnKk.json")
    if not cache_file.exists():
        print("Run fetcher.py first to download the transcript.")
    else:
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        chunks = create_chunks(data)

        # 1. Initialize store & index chunks
        store = VectorStore()
        store.store_chunks(chunks)

        # 2. Test semantic search!
        test_question = "What is a neuron and how does it hold a number?"
        print(f"\n🔍 Searching for: '{test_question}'")
        hits = store.search(test_question, top_k=2)

        print("\n🏆 Top 2 Most Relevant Chunks:")
        for i, hit in enumerate(hits, 1):
            print(f"\n--- Hit {i} (Similarity: {hit['similarity']}) ---")
            print(f"Timestamp: {hit['start']}s - {hit['end']}s (https://youtu.be/{hit['video_id']}?t={int(hit['start'])})")
            print(f"Text snippet: {hit['text'][:250]}...")
