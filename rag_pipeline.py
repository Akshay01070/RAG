"""
RAG Pipeline Module
===================
Handles document chunking, embedding, FAISS indexing, retrieval, and LLM-based answer generation.

Embedding Model : sentence-transformers/all-MiniLM-L6-v2  (384-dim, local, free)
Vector Store    : FAISS (flat L2 index)
LLM             : OpenRouter API (free-tier model)
"""

import os
import json
import glob
import re
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384
DOCUMENTS_DIR = os.path.join(os.path.dirname(__file__), "documents")
VECTOR_STORE_DIR = os.path.join(os.path.dirname(__file__), "vector_store")
FAISS_INDEX_PATH = os.path.join(VECTOR_STORE_DIR, "index.faiss")
CHUNKS_META_PATH = os.path.join(VECTOR_STORE_DIR, "chunks_meta.json")
TOP_K = 5
CHUNK_SIZE = 500          # approximate characters per chunk
CHUNK_OVERLAP = 100       # overlap between consecutive chunks

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_MODELS = [
    "meta-llama/llama-3.3-70b-instruct:free",
    "google/gemma-3-27b-it:free",
    "nvidia/llama-3.1-nemotron-nano-8b-v1:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "openrouter/auto",
]
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds (doubles each retry)

# ---------------------------------------------------------------------------
# Singleton embedding model loader
# ---------------------------------------------------------------------------
_embed_model = None

def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        print("[RAG] Loading embedding model …")
        _embed_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print("[RAG] Embedding model loaded.")
    return _embed_model

# ---------------------------------------------------------------------------
# 1. Document Chunking
# ---------------------------------------------------------------------------

def _load_documents():
    """Load all .txt documents from the documents directory."""
    docs = []
    for fpath in sorted(glob.glob(os.path.join(DOCUMENTS_DIR, "*.txt"))):
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({
            "filename": os.path.basename(fpath),
            "text": text,
        })
    return docs


def _chunk_text(text: str, source: str) -> list[dict]:
    """
    Split text into overlapping chunks using a paragraph-aware strategy.

    Strategy:
    1. Split on double-newlines to get paragraphs.
    2. Merge small paragraphs together until CHUNK_SIZE is reached.
    3. Apply CHUNK_OVERLAP by carrying trailing characters into the next chunk.
    """
    # Split into paragraphs / sections
    paragraphs = re.split(r"\n{2,}", text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []
    current_chunk = ""
    chunk_id = 0

    for para in paragraphs:
        # If adding this paragraph exceeds the size, save current and start new
        if current_chunk and len(current_chunk) + len(para) + 2 > CHUNK_SIZE:
            chunks.append({
                "id": f"{source}::chunk_{chunk_id}",
                "source": source,
                "text": current_chunk.strip(),
            })
            chunk_id += 1
            # Overlap: carry tail of old chunk
            overlap_text = current_chunk[-CHUNK_OVERLAP:] if len(current_chunk) > CHUNK_OVERLAP else current_chunk
            current_chunk = overlap_text + "\n\n" + para
        else:
            current_chunk = (current_chunk + "\n\n" + para) if current_chunk else para

    # Last chunk
    if current_chunk.strip():
        chunks.append({
            "id": f"{source}::chunk_{chunk_id}",
            "source": source,
            "text": current_chunk.strip(),
        })

    return chunks


def chunk_all_documents() -> list[dict]:
    """Load and chunk every document in the documents directory."""
    docs = _load_documents()
    all_chunks = []
    for doc in docs:
        all_chunks.extend(_chunk_text(doc["text"], doc["filename"]))
    print(f"[RAG] Chunked {len(docs)} documents into {len(all_chunks)} chunks.")
    return all_chunks

# ---------------------------------------------------------------------------
# 2. Embedding & FAISS Index
# ---------------------------------------------------------------------------

def build_index(chunks: list[dict] | None = None):
    """Embed all chunks and build a FAISS index, persisting to disk."""
    if chunks is None:
        chunks = chunk_all_documents()

    model = _get_embed_model()
    texts = [c["text"] for c in chunks]
    print("[RAG] Generating embeddings …")
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(EMBEDDING_DIM)   # Inner-product (cosine after normalization)
    index.add(embeddings)

    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    faiss.write_index(index, FAISS_INDEX_PATH)
    with open(CHUNKS_META_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    print(f"[RAG] FAISS index built with {index.ntotal} vectors → {FAISS_INDEX_PATH}")
    return index, chunks


def load_index():
    """Load persisted FAISS index and chunk metadata from disk."""
    if not os.path.exists(FAISS_INDEX_PATH):
        print("[RAG] No existing index found — building from documents …")
        return build_index()
    index = faiss.read_index(FAISS_INDEX_PATH)
    with open(CHUNKS_META_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"[RAG] Loaded FAISS index with {index.ntotal} vectors.")
    return index, chunks

# ---------------------------------------------------------------------------
# 3. Retrieval
# ---------------------------------------------------------------------------

def retrieve(query: str, index, chunks: list[dict], top_k: int = TOP_K) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a query."""
    model = _get_embed_model()
    q_emb = model.encode([query], normalize_embeddings=True).astype("float32")
    distances, indices = index.search(q_emb, top_k)

    results = []
    for rank, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < 0:
            continue
        chunk = chunks[idx].copy()
        chunk["score"] = float(dist)
        chunk["rank"] = rank + 1
        results.append(chunk)

    return results

# ---------------------------------------------------------------------------
# 4. LLM Answer Generation (with strict grounding + fallback)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an AI assistant for Indecimal, a construction marketplace.

STRICT RULES:
1. Answer the user's question ONLY using the CONTEXT provided below.
2. Do NOT use any outside knowledge, assumptions, or speculation.
3. If the context does not contain enough information to answer the question, respond EXACTLY with:
   "I couldn't find an answer to that in the provided documents."
4. When quoting prices, brands, or specifications, use the exact values from the context.
5. Cite which document or section the information comes from when possible.
6. Keep your answer clear, concise, and well-structured.
"""

def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Call the OpenRouter LLM with retrieved context and return the answer.
    
    Includes automatic retry with exponential backoff and fallback across
    multiple free models to handle rate-limiting (429 errors).
    """
    import time

    # Build context string
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source_label = chunk.get("source", "unknown")
        context_parts.append(f"--- Context Chunk {i} (from {source_label}) ---\n{chunk['text']}")
    context_str = "\n\n".join(context_parts)

    user_message = f"""CONTEXT:
{context_str}

USER QUESTION:
{query}

Answer the question using ONLY the context above. If the context does not contain the answer, say "I couldn't find an answer to that in the provided documents."
"""

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "http://localhost:5000",
        "X-Title": "Indecimal RAG Assistant",
    }

    last_error = None

    for model_id in LLM_MODELS:
        delay = RETRY_DELAY
        for attempt in range(1, MAX_RETRIES + 1):
            payload = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                "temperature": 0.2,
                "max_tokens": 1024,
            }
            try:
                print(f"[RAG] Trying {model_id} (attempt {attempt}/{MAX_RETRIES}) …")
                resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
                answer = data["choices"][0]["message"]["content"].strip()
                print(f"[RAG] ✓ Answer generated via {model_id}")
                return answer
            except requests.exceptions.HTTPError as e:
                status_code = resp.status_code
                last_error = str(e)
                if status_code == 429:
                    print(f"[RAG] 429 rate-limited on {model_id}, retrying in {delay}s …")
                    time.sleep(delay)
                    delay *= 2  # exponential backoff
                    continue
                else:
                    # Non-retryable HTTP error — try next model
                    try:
                        last_error = str(resp.json())
                    except Exception:
                        last_error = resp.text
                    print(f"[RAG] HTTP {status_code} on {model_id}: {last_error}")
                    break
            except Exception as e:
                last_error = str(e)
                print(f"[RAG] Error on {model_id}: {last_error}")
                break
        # If we exhausted retries for this model, move to next
        print(f"[RAG] Exhausted retries for {model_id}, trying next model …")

    return f"[LLM Error] All models failed. Last error: {last_error}"

# ---------------------------------------------------------------------------
# 5. Full RAG Query (convenience)
# ---------------------------------------------------------------------------

def query_rag(query: str, index=None, chunks=None, top_k: int = TOP_K) -> dict:
    """End-to-end: retrieve context → generate answer → return both."""
    if index is None or chunks is None:
        index, chunks = load_index()

    retrieved = retrieve(query, index, chunks, top_k)
    answer = generate_answer(query, retrieved)

    return {
        "query": query,
        "answer": answer,
        "sources": [
            {
                "rank": r["rank"],
                "score": round(r["score"], 4),
                "source": r["source"],
                "text": r["text"],
            }
            for r in retrieved
        ],
    }
