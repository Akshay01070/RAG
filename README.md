# Mini RAG — Indecimal Construction Marketplace Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about Indecimal's construction services using internal documents. All answers are **grounded strictly in retrieved content** — if the documents don't contain the answer, the bot says so.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![Flask](https://img.shields.io/badge/Flask-3.1-green) ![FAISS](https://img.shields.io/badge/FAISS-Vector%20Search-orange) ![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 🏗️ Architecture Overview

```
User Query
    │
    ▼
┌──────────────┐     ┌──────────────────────┐     ┌───────────────┐
│  Flask API   │────▶│  Embedding Model     │────▶│  FAISS Index  │
│  /api/query  │     │  (all-MiniLM-L6-v2)  │     │  (L2 Search)  │
└──────────────┘     └──────────────────────┘     └───────┬───────┘
                                                          │ top-k chunks
                                                          ▼
                                                  ┌───────────────┐
                                                  │  OpenRouter    │
                                                  │  LLM (Llama)  │
                                                  │  + Grounding   │
                                                  │    Prompt      │
                                                  └───────┬───────┘
                                                          │
                                                          ▼
                                               Answer + Source Chunks
                                               (displayed in chatbot)
```

---

## 🔧 Setup & Run Locally

### Prerequisites
- Python 3.10+
- An OpenRouter API key (free at [openrouter.ai/keys](https://openrouter.ai/keys))

### Installation

```bash
# 1. Clone the repo
git clone https://github.com/your-username/mini-rag.git
cd mini-rag

# 2. Create a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your API key
copy .env.example .env
# Edit .env and paste your OpenRouter API key

# 5. Run the app
python app.py
```

Open **http://localhost:5000** in your browser.

---

## 📄 Documents Used

| # | Document | Content |
|---|----------|---------|
| 1 | `doc1_company_overview.txt` | Company overview, operating principles, customer journey, FAQs |
| 2 | `doc2_package_specs.txt` | Package pricing (Essential → Pinnacle), materials, wallet allowances |
| 3 | `doc3_customer_protection.txt` | Payment safety, delay management, quality assurance, maintenance |

---

## 🧠 Model Choices & Rationale

### Embedding Model — `all-MiniLM-L6-v2`
- **Why**: Produces 384-dimensional embeddings with excellent semantic similarity performance. It's the most popular sentence-transformer model — fast, accurate, and runs locally with no API key.
- **Alternative considered**: OpenAI embeddings — rejected because they require a paid API and add latency.

### LLM — `meta-llama/llama-3.1-8b-instruct:free` (via OpenRouter)
- **Why**: Free tier on OpenRouter, strong instruction-following ability, good at generating structured answers from context.
- **Grounding**: The system prompt explicitly instructs the model to answer **only** from retrieved chunks and to fall back to *"I couldn't find an answer to that in the provided documents"* when context is insufficient.

---

## ✂️ Document Chunking Strategy

**Method**: Paragraph-aware chunking with configurable overlap.

1. Each document is split on double-newlines (`\n\n`) to respect paragraph boundaries.
2. Small paragraphs are merged until the chunk reaches ~500 characters.
3. An overlap of ~100 characters carries context from the end of one chunk into the next, preventing information loss at chunk boundaries.

**Why not sentence-level?** Paragraph-aware chunking preserves the logical structure of headings, lists, and specifications — critical for construction documents with nested data (e.g., package comparisons).

---

## 🔍 Retrieval — FAISS Vector Search

- **Index type**: `IndexFlatIP` (Inner Product on normalized vectors = cosine similarity).
- **Top-k**: 5 chunks are retrieved per query.
- **Embedding normalization**: Both document and query embeddings are L2-normalized before indexing, making IP equivalent to cosine similarity.
- **Persistence**: Index and chunk metadata are saved to `vector_store/` on disk and loaded at startup for instant queries.

---

## 🛡️ Grounding Enforcement

Three layers ensure the model stays grounded:

1. **System Prompt**: Explicitly states "Answer ONLY using the CONTEXT provided. Do NOT use outside knowledge."
2. **User Prompt Structure**: Context chunks are injected before the question, followed by a repeated instruction to use only the context.
3. **Strict Fallback**: If context is insufficient, the model must respond with: *"I couldn't find an answer to that in the provided documents."*

---

## 📊 Quality Analysis — Test Questions & Evaluation

We tested 12 questions derived from the 3 documents. Each was evaluated for:
- **Retrieval Relevance**: Were the top-k chunks relevant to the question?
- **Hallucination**: Did the answer contain unsupported claims?
- **Completeness**: Did the answer cover the key information?

### Test Results

| # | Question | Retrieval | Hallucination | Completeness | Notes |
|---|----------|-----------|---------------|--------------|-------|
| 1 | What factors affect construction project delays? | ✅ Relevant | ✅ None | ✅ Complete | Correctly cited delay management mechanisms |
| 2 | What are Indecimal's package prices per sqft? | ✅ Relevant | ✅ None | ✅ Complete | Listed all 4 packages with prices |
| 3 | What steel brands are used in the Pinnacle package? | ✅ Relevant | ✅ None | ✅ Complete | Correctly returned TATA, ₹80,000/MT |
| 4 | How does Indecimal handle contractor payments? | ✅ Relevant | ✅ None | ✅ Complete | Described escrow + stage-based model |
| 5 | What is included in the zero cost maintenance program? | ✅ Relevant | ✅ None | ✅ Complete | Listed all coverage areas |
| 6 | What type of cement is used in the Premier package? | ✅ Relevant | ✅ None | ✅ Complete | Dalmia/Bharthi, ₹370/bag |
| 7 | How does the escrow payment model work? | ✅ Relevant | ✅ None | ✅ Complete | Described 3-step escrow flow |
| 8 | What are the window specifications for the Infinia package? | ✅ Relevant | ✅ None | ✅ Complete | UPVC, ₹600/sqft, 3-track with mesh |
| 9 | What painting brands are used for exterior walls? | ✅ Relevant | ✅ None | ✅ Complete | Listed all 4 packages' exterior paint brands |
| 10 | How many quality checkpoints does Indecimal use? | ✅ Relevant | ✅ None | ✅ Complete | 445+ checkpoints across lifecycle |
| 11 | What is the financing timeline for home loans? | ✅ Relevant | ✅ None | ⚠️ Partial | Mentioned ~7 days / ~30 days with T&C caveat |
| 12 | What is Indecimal's policy on real-time tracking? | ✅ Relevant | ✅ None | ✅ Complete | Dashboard, live photo updates, app tracking |

### Observations

1. **Retrieval quality is strong** — paragraph-aware chunking ensures that related specifications (e.g., all 4 packages' steel brands) stay in the same chunk, making retrieval more complete.
2. **No hallucinations detected** — the strict grounding prompt + fallback mechanism effectively prevent the model from inventing information.
3. **Partial completeness on Q11** — financing details are spread across two documents; retrieval captured the main chunk but a secondary mention was ranked lower.
4. **Edge case**: Questions about topics not in the documents (e.g., "What is the weather in Bangalore?") correctly trigger the fallback response.
5. **Overlap chunking helps** — the 100-character overlap prevents edge cases where critical information sits at a chunk boundary.

---

## 📁 Project Structure

```
mini-rag/
├── app.py                  # Flask web server
├── rag_pipeline.py         # Core RAG: chunking, embedding, FAISS, LLM
├── requirements.txt        # Python dependencies
├── .env.example            # API key template
├── .gitignore
├── README.md
├── documents/              # Source documents
│   ├── doc1_company_overview.txt
│   ├── doc2_package_specs.txt
│   └── doc3_customer_protection.txt
├── vector_store/            # Generated FAISS index (auto-created)
│   ├── index.faiss
│   └── chunks_meta.json
└── static/                  # Frontend
    ├── index.html
    ├── style.css
    └── script.js
```

---

## 📜 License

MIT
