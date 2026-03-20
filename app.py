"""
Flask Application — Indecimal RAG Assistant
============================================
Serves the chatbot frontend and exposes the /api/query endpoint.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import rag_pipeline

app = Flask(__name__, static_folder="static", static_url_path="")
CORS(app)

# Pre-load FAISS index at startup
print("[APP] Initialising RAG pipeline …")
_index, _chunks = rag_pipeline.load_index()
print(f"[APP] Ready — {len(_chunks)} chunks indexed.")

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def serve_frontend():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/query", methods=["POST"])
def api_query():
    """Accept a user query and return the RAG-generated answer + sources."""
    data = request.get_json(force=True)
    user_query = data.get("query", "").strip()

    if not user_query:
        return jsonify({"error": "Empty query"}), 400

    result = rag_pipeline.query_rag(user_query, index=_index, chunks=_chunks)
    return jsonify(result)


@app.route("/api/reindex", methods=["POST"])
def api_reindex():
    """Rebuild the FAISS index from the documents directory."""
    global _index, _chunks
    _index, _chunks = rag_pipeline.build_index()
    return jsonify({"status": "ok", "total_chunks": len(_chunks)})


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
