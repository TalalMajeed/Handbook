# System Design — Handbook QA System

A scalable Question-Answering system over the **NUST SEECS UG & PG Student Handbooks**,
built using Big Data retrieval techniques: TF-IDF, MinHash+LSH, SimHash, PageRank,
sentence embeddings, and LLM-based answer generation.

---

## Table of Contents

1. [High-Level Architecture](#1-high-level-architecture)
2. [Two-Phase Design](#2-two-phase-design)
3. [Project Structure](#3-project-structure)
4. [File-by-File Explanation](#4-file-by-file-explanation)
5. [Data Flow Diagram](#5-data-flow-diagram)
6. [Key Design Decisions](#6-key-design-decisions)
7. [API Endpoints](#7-api-endpoints)
8. [Dependencies](#8-dependencies)

---

## 1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HANDBOOK QA SYSTEM                          │
│                                                                      │
│  ┌──────────────┐     ┌──────────────────────────────────────────┐  │
│  │  Data Layer  │     │            Retrieval Layer               │  │
│  │              │     │                                          │  │
│  │ ug-handbook  │────▶│  TF-IDF  │  MinHash+LSH  │  Semantic    │  │
│  │ pg-handbook  │     │  (exact) │  (approx.)    │  Embeddings  │  │
│  │   (PDFs)     │     │          └──────────────────────────────-│  │
│  └──────────────┘     │               PageRank Reranking         │  │
│                        └──────────────────┬───────────────────────┘  │
│                                           │                          │
│  ┌──────────────────────────────────────┐ │                          │
│  │         Generation Layer             │◀┘                          │
│  │                                      │                            │
│  │  Groq Llama-3.1-8b  →  flan-t5-base │                            │
│  │  (primary)             (fallback 1)  │                            │
│  │              →  Extractive (fallback2│                            │
│  └──────────────────┬───────────────────┘                            │
│                     │                                                │
│  ┌──────────────────▼───────────────────┐                            │
│  │         Presentation Layer           │                            │
│  │                                      │                            │
│  │  FastAPI Server  +  SSE Streaming    │                            │
│  │  Chat UI  +  Evidence Panel          │                            │
│  └──────────────────────────────────────┘                            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Two-Phase Design

### Phase 1 — Offline Indexing (`build_index.py`)
Run **once** to build the search index. Takes ~2–3 minutes on CPU.

```
PDF files
   │
   ▼
[Step 1]  Load pages via PyMuPDF (fitz)
   │
   ▼
[Step 2]  Strip running headers & page numbers (strip_headers)
   │
   ▼
[Step 3]  Sentence-aware chunking (50–250 words, 1-sentence overlap)
   │
   ▼
[Step 4]  SimHash deduplication (64-bit fingerprint, Hamming ≤ 3)
   │
   ▼
[Step 5]  Compute sentence embeddings (all-MiniLM-L6-v2, 384-dim)
   │
   ├──▶ [Step 6a]  Build PageRank graph (cosine sim > 0.65)
   ├──▶ [Step 6b]  Build TF-IDF index (15k features, synonyms, bigrams)
   └──▶ [Step 6c]  Build MinHash+LSH index (128 hashes, 16 bands)
            │
            ▼
   Save to handbook-1.0/:
     chunks.pkl   tfidf.pkl   lsh.pkl   embed_model/
```

### Phase 2 — Online Query (`server.py` + `pipeline.py`)
Runs on every user query. Real-time, streamed response.

```
User types question in browser
   │
   ▼
[Step 1]  _clean()         — normalise, preserve "2.75", "75%"
   │
   ▼
[Step 2]  _expand_query()  — GPA↔CGPA, fail↔repeat synonym expansion
   │
   ▼
[Step 3]  TF-IDF pre-filter (top-40 candidates, gate score > 0.005)
   │
   ▼
[Step 4]  Semantic scoring (sentence embedding cosine similarity)
   │
   ▼
[Step 5]  Hybrid score = 0.70×TF-IDF + 0.28×Semantic + 0.02×PageRank
   │
   ▼
[Step 6]  Top-5 chunks → Groq API (llama-3.1-8b-instant)
   │
   ▼
[Step 7]  Stream answer tokens → browser via SSE
   │
   ▼
[Step 8]  Append citations → emit sources event → update evidence panel
```

---

## 3. Project Structure

```
Handbook/
│
├── data/                         ← Input PDFs (not committed to Git)
│   ├── ug-handbook.pdf           ← NUST SEECS Undergraduate Handbook
│   └── pg-handbook.pdf           ← NUST SEECS Postgraduate Handbook
│
├── interface/                    ← Web UI (served as static files)
│   ├── index.html                ← Chat interface (two-panel layout)
│   ├── styles.css                ← Dark theme, chat bubbles, modals
│   ├── logic.js                  ← SSE client, evidence panel, modal
│   └── workflow.html             ← Animated pipeline walkthrough page
│
├── handbook-1.0/                 ← Pre-built index artifacts (generated)
│   ├── chunks.pkl                ← All deduplicated chunks + embeddings + PageRank
│   ├── tfidf.pkl                 ← TfidfVectorizer + sparse TF-IDF matrix
│   ├── lsh.pkl                   ← MinHasher + signature matrix + LSHIndex
│   └── embed_model/              ← Saved all-MiniLM-L6-v2 weights
│
├── build_index.py                ← OFFLINE: Full indexing pipeline
├── pipeline.py                   ← ONLINE: 3 retrieval methods
├── lsh_core.py                   ← Shared MinHasher + LSHIndex classes
├── server.py                     ← FastAPI server + generation + SSE
├── experiments.py                ← Benchmarking script (15 queries × 3 methods)
├── analysis.ipynb                ← Jupyter notebook (exploratory prototype)
├── check_files.py                ← Utility: verify artifact files exist
├── requirements.txt              ← Python dependencies
├── .env                          ← GROQ_API_KEY (never commit to Git)
├── .gitignore                    ← Excludes .env, handbook-1.0/, data/
└── README.md                     ← Setup and usage instructions
```

---

## 4. File-by-File Explanation

---

### `build_index.py` — Offline Indexing Pipeline

**Purpose:** Reads both PDF handbooks and builds all search indexes. Run once before starting the server.

**What is implemented:**

| Function | What it does |
|---|---|
| `load_pdf(path)` | Opens a PDF with PyMuPDF, extracts raw text per page, attaches page number and source filename |
| `strip_headers(text)` | Removes running headers (`"NUST Undergraduate Student Handbook 87"`), standalone page numbers, and school name stamps using regex |
| `split_sentences(text)` | Splits text on `.!?` followed by uppercase/digit — simple sentence boundary detector |
| `chunk_pages(pages)` | Sentence-aware chunker: accumulates sentences up to 250 words, emits chunk, keeps 1-sentence overlap for next chunk |
| `compute_simhash(text)` | Produces a 64-bit SimHash fingerprint by projecting MD5 token hashes into a weighted bit-accumulator |
| `hamming_distance(h1, h2)` | Returns bit difference between two SimHash fingerprints via XOR + popcount |
| `simhash_dedup(chunks)` | Removes near-duplicate chunks with Hamming distance ≤ 3 |
| `compute_embeddings(chunks, model)` | Encodes all chunk texts with `all-MiniLM-L6-v2`, stores 384-dim normalized vectors in each chunk dict |
| `compute_pagerank(chunks, embeddings)` | Builds a NetworkX graph (edges where cosine sim > 0.65), runs PageRank (damping=0.85), stores score per chunk |
| `build_tfidf(chunks)` | Fits `TfidfVectorizer` on synonym-expanded chunk texts (15k features, bigrams, sublinear TF, domain stop-words) |
| `expand_synonyms(text)` | Appends synonym tokens to chunk text for TF-IDF fitting (e.g. CGPA → "GPA grade point average") |
| `build_lsh(chunks)` | Creates MinHasher + LSHIndex, computes 128-element signatures for all chunks, inserts into 16-band LSH index |
| `save_artifacts(...)` | Pickles chunks, TF-IDF model, LSH index to `handbook-1.0/`; saves embedding model |
| `main()` | Orchestrates all 8 steps in order with progress printing |

**Run:** `python build_index.py`
**Output:** `handbook-1.0/` directory with 4 artifact files

---

### `lsh_core.py` — MinHash + LSH Core Classes

**Purpose:** Standalone module containing the `MinHasher` and `LSHIndex` classes. Kept separate so both `build_index.py` and `pipeline.py` can import it, and `pickle` can resolve class references correctly.

**What is implemented:**

| Class / Method | What it does |
|---|---|
| `MinHasher.__init__` | Generates `n_hashes` random `(a, b)` pairs for universal hash family `h(x) = (a·x + b) mod p` using Mersenne prime p = 2^61−1 |
| `MinHasher._hash_token(token)` | Maps a string token to an integer via MD5, returns `int mod p` |
| `MinHasher.get_shingles(text, k=3)` | Returns the set of all word k-grams (3-shingles) from the text |
| `MinHasher.signature(text)` | Computes the 128-element MinHash signature: for each hash function, takes the minimum hash value across all shingles |
| `LSHIndex.__init__` | Creates `n_bands` empty bucket dictionaries. Validates `n_hashes % n_bands == 0` |
| `LSHIndex.threshold` | Property: returns `(1/n_bands)^(1/rows_per_band)` — the Jaccard collision probability = 0.5 threshold |
| `LSHIndex._band_key(sig, band)` | Hashes the band's sub-signature slice into a single bucket key |
| `LSHIndex.add(idx, sig)` | Inserts a chunk index into its band buckets |
| `LSHIndex.query(sig)` | Returns the union of all chunk indices sharing at least one band bucket with the query signature |

**Constants:** `N_HASHES=128`, `N_BANDS=16`, `SHINGLE_SIZE=3`

---

### `pipeline.py` — Retrieval Pipeline

**Purpose:** Loads pre-built index artifacts and exposes three retrieval methods used by the server at query time.

**What is implemented:**

| Method | What it does |
|---|---|
| `HandbookPipeline.__init__(path)` | Loads `chunks.pkl`, `tfidf.pkl`, `lsh.pkl`, and saved embedding model from `handbook-1.0/` |
| `_clean(query)` | Lowercases, preserves `.`, `-`, `%` (important for "2.75", "75%"), removes other special chars |
| `_expand_query(query)` | Applies synonym expansion to the query (mirrors index-time expansion) so "GPA" matches "CGPA" chunks |
| `_semantic_score(q_emb, chunk)` | Returns dot product of query embedding and chunk embedding (both L2-normalized = cosine similarity) |
| `retrieve_tfidf(query, top_k=8)` | **Exact baseline:** expands query, transforms with TF-IDF vectorizer, computes sparse cosine similarity against all chunks, returns top-k by score |
| `retrieve_lsh(query, top_k=8)` | **Approximate:** computes query MinHash signature, queries LSH buckets for candidates, re-ranks by Jaccard similarity. Falls back to linear MinHash scan if no bucket matches |
| `retrieve(query, top_k=8)` | **Hybrid (production):** TF-IDF pre-filter (top-40, gate > 0.005) → semantic scoring on cleaned query → hybrid score (0.70 TF + 0.28 SEM + 0.02 PR) → return top-k |
| `benchmark(query, top_k=8)` | Runs all three retrievers with timing, returns results dict for experiment comparison |

**Key constants:** `ALPHA_TFIDF=0.70`, `ALPHA_SEMANTIC=0.28`, `ALPHA_PAGERANK=0.02`, `MIN_TFIDF_SCORE=0.005`, `TFIDF_PRE_K=40`, `CHUNK_TOP_K=8`

---

### `server.py` — FastAPI Web Server

**Purpose:** Serves the web UI, handles query requests, runs retrieval via `HandbookPipeline`, generates answers via Groq API (or fallback), and streams responses to the browser using Server-Sent Events (SSE).

**What is implemented:**

| Function / Route | What it does |
|---|---|
| Startup | Loads `.env` for `GROQ_API_KEY`, initialises Groq client if key present, loads flan-t5-base if Groq unavailable, loads `HandbookPipeline` |
| `sse_event(event, data)` | Formats a string as an SSE `event: ... / data: ...` frame |
| `format_chunk_for_client(chunk, rank)` | Extracts rank, text, preview (280 chars), page, source, and all 4 scores into a JSON-serializable dict |
| `build_context_for_model(docs)` | Assembles clean numbered context blocks `[Chunk 1]\n{text}` with no inline source tags |
| `build_citations(docs)` | Produces `Sources: [ug-handbook, p.25]  [pg-handbook, p.12]` string appended after the answer |
| `is_answer_valid(text)` | Returns False if answer is too short, starts with "not found", echoes "chunk", etc. |
| `extractive_fallback(query, docs)` | Pure keyword-overlap sentence extractor — finds best sentences from top chunks with no model required |
| `generate_with_groq(query, docs)` | Streams answer tokens from Groq API (`llama-3.1-8b-instant`, temp=0.1, max_tokens=1024) |
| `generate_with_flanT5(query, docs)` | Local generation with flan-t5-base (top-3 chunks, 120 words/chunk, deterministic beam search) |
| `stream_generation(prompt)` | Main SSE generator: retrieves docs → generates answer → emits `start`, `delta`, `replace`, `sources`, `done` events |
| `GET /` | Returns `interface/index.html` |
| `GET /workflow` | Returns `interface/workflow.html` |
| `GET /retrieve` | Returns top-8 chunks as JSON (used by the evidence panel) |
| `GET /generate` | Returns `StreamingResponse` with SSE stream for full retrieve+generate pipeline |

---

### `interface/index.html` — Chat Interface

**Purpose:** Main user-facing page. Two-column layout with chat on the left and retrieved evidence on the right.

**What is implemented:**
- Top bar with status dot (Ready / Connecting / Streaming / Error), title, Clear button, Workflow link
- **Left panel:** scrollable chat log with user and assistant message bubbles, auto-resize textarea composer, send button
- **Right panel:** evidence list showing retrieved chunk cards (rank, source, page, score)
- **Modal:** full-screen overlay for reading complete chunk text with all score badges
- Links `styles.css` and `logic.js`

---

### `interface/styles.css` — Dark Theme Stylesheet

**Purpose:** All visual styling for the chat UI.

**What is implemented:**
- CSS custom properties (design tokens): `--bg`, `--surface`, `--accent`, `--text-muted`, etc.
- Topbar, status dot animations (pulse for connecting/streaming)
- Two-column flex layout (chat panel fills remaining width, evidence panel fixed 380px)
- Chat message bubbles (user = blue-tinted, assistant = dark surface, system = dashed border)
- Typing indicator (three bouncing dots animation)
- Chunk card styles with hover lift effect and fade-in animation
- Modal backdrop blur + scale-in animation
- Responsive: evidence panel hidden on screens < 800px

---

### `interface/logic.js` — Client-Side JavaScript

**Purpose:** All browser-side logic — form submission, SSE stream handling, evidence panel rendering, modal management.

**What is implemented:**

| Function | What it does |
|---|---|
| `createMessage(role, content)` | Appends a new chat bubble to the log |
| `createTyping()` | Appends the three-dot typing indicator |
| `setStatus(state, title)` | Updates the status dot class (ready/connecting/streaming/error) |
| `setBusy(bool)` | Disables/enables the send button, textarea, and clear button |
| `openModal(chunk)` | Populates and shows the chunk detail modal with full text and all 4 score badges |
| `closeModal()` | Hides modal, restores body scroll |
| `renderEvidence(chunks)` | Renders all chunk cards into the evidence panel with rank, source, page, preview, and click handler |
| `stopStream()` | Closes the active `EventSource` connection |
| Form submit handler | Creates SSE connection to `/generate`, wires `start`, `delta`, `replace`, `sources`, `done`, `error` event listeners |
| `autoResize()` | Grows textarea height up to 160px as user types |
| Keyboard shortcuts | Enter to send (Shift+Enter = newline), arrow keys handled by modal |

---

### `interface/workflow.html` — Animated Pipeline Walkthrough

**Purpose:** Standalone page (`/workflow`) that visually explains the system's 8-step indexing pipeline and 5-step query pipeline with animated timeline cards.

**What is implemented:**
- Tab switcher: "Phase 1 — Building the Library" and "Phase 2 — Answering a Query"
- Animated vertical timeline with step cards that slide in with opacity transition
- Each card: expandable with detail grids, analogy boxes, code snippets, tech tags
- Run Walkthrough button with configurable speed (Slow / Normal / Fast)
- Progress bar tracking animation completion
- Full self-contained CSS and JavaScript (no external dependencies)

---

### `experiments.py` — Benchmarking Script

**Purpose:** Formal experimental analysis required for the project report. Compares all three retrieval methods across 15 standard queries and measures scalability.

**What is implemented:**

| Function | What it does |
|---|---|
| `precision_at_k(texts, keywords, k=8)` | Counts how many of the top-k retrieved chunks contain at least one ground-truth keyword |
| `experiment_retrieval_comparison(p)` | Runs all 3 retrievers on 15 queries, prints latency and Precision@8 per query and average |
| `experiment_parameter_sensitivity()` | Tests effect of n_hashes (32/64/128/256), n_bands (4/8/16/32/64), and SimHash threshold (0/1/2/3/5/8) |
| `experiment_scalability()` | Duplicates corpus 1×/2×/5×/10×, measures TF-IDF query time vs LSH query time at each scale |

**15 benchmark queries cover:** GPA, course failure, attendance, course repeat, credit limits, graduation, grading scale, plagiarism, CGPA calculation, academic probation, electives, fee refund, incomplete grades, credit transfer, thesis submission.

---

### `analysis.ipynb` — Jupyter Prototype Notebook

**Purpose:** Early exploratory notebook used during development to prototype the pipeline step-by-step before it was refactored into production Python files.

**What is implemented:**
- Basic PDF loading and chunking (early version without header stripping)
- Prototype SimHash using Python's built-in `hash()` (not production-grade)
- Sentence embedding with `all-MiniLM-L6-v2`
- NetworkX PageRank construction
- Basic TF-IDF vectorization
- Artifact saving with pickle

> **Note:** This notebook represents the prototype. All production code is in the `.py` files.

---

### `check_files.py` — Artifact Verification Utility

**Purpose:** Quick sanity check to confirm that `build_index.py` ran successfully and all required artifacts exist.

**What is implemented:** Checks for existence of `chunks.pkl`, `tfidf.pkl`, `lsh.pkl`, and `embed_model/` in `handbook-1.0/`. Prints size and chunk count if found, error message if missing.

---

### `requirements.txt` — Python Dependencies

| Package | Purpose |
|---|---|
| `fastapi`, `uvicorn` | Web server and ASGI runner |
| `pymupdf` | PDF text extraction (imported as `fitz`) |
| `sentence-transformers` | `all-MiniLM-L6-v2` neural embeddings |
| `scikit-learn` | `TfidfVectorizer`, cosine similarity |
| `networkx` | PageRank graph computation |
| `transformers`, `torch` | Local `flan-t5-base` generation fallback |
| `accelerate` | Transformer model loading optimisation |
| `numpy` | Array operations, MinHash computations |
| `groq` | Groq API client for Llama-3.1 generation |
| `python-dotenv` | Reads `GROQ_API_KEY` from `.env` file |

---

### `.env` — API Key Storage

**Purpose:** Stores the Groq API key so it is never hardcoded in source files.

```
GROQ_API_KEY=gsk_your_key_here
```

> ⚠️ Listed in `.gitignore` — never committed to version control.

---

## 5. Data Flow Diagram

```
User Query: "What is the minimum GPA requirement?"
    │
    ▼
_clean()  →  "what is the minimum gpa requirement"
    │
    ▼
_expand_query()  →  adds "cgpa grade point average undergraduate"
    │
    ▼
TF-IDF vectorizer.transform([expanded_query])
    │
    ▼
scores = tfidf_matrix @ q_vec.T   [317 × 1 sparse vector]
    │
    ▼
Filter: scores > 0.005  →  ~40 candidate chunk indices
    │
    ▼
embed_model.encode([cleaned_query])  →  q_emb [384-dim]
    │
    ▼
For each candidate i:
    sem     = dot(q_emb, chunks[i]["embedding"])
    tfidf   = scores[i]
    pr      = chunks[i]["pagerank"]
    hybrid  = 0.70*tfidf + 0.28*sem + 0.02*pr
    │
    ▼
Sort by hybrid score → top-8 chunks returned
    │
    ▼
Top-5 chunks → build_context_for_model()
    │
    ▼
Groq API: llama-3.1-8b-instant (stream=True, temp=0.1)
    │
    ▼
For each token delta:
    yield sse_event("delta", token)  →  browser appends to chat bubble
    │
    ▼
After stream:
    yield sse_event("sources", JSON chunks)  →  evidence panel populated
    yield sse_event("done", "finished")
```

---

## 6. Key Design Decisions

| Decision | Why |
|---|---|
| **Sentence-aware chunking over blind word windows** | Blind 300-word windows split policy rules mid-sentence. Sentence boundaries ensure each chunk contains complete, self-contained rules |
| **Synonym expansion at both index and query time** | TF-IDF is purely bag-of-words. Without expansion, "GPA" never matches "CGPA" chunks. Symmetric expansion fixes this at zero runtime cost |
| **MIN_TFIDF_SCORE = 0.005 gate** | Prevents high-PageRank but query-irrelevant chunks from entering the semantic scoring stage |
| **PageRank weight reduced 0.15 → 0.02** | At 0.15, chapters with many cross-references ranked above specific policy chunks. At 0.02 it acts only as a tiebreaker |
| **Semantic scoring on cleaned query, not expanded** | Synonym noise in the expanded string (e.g. "cgpa gpa grade point") degrades the sentence embedding quality |
| **Source tags stripped from LLM context** | When `[ug-handbook, p.25]` was inside the context string, the model echoed it verbatim into answers |
| **Groq API instead of local flan-t5-base** | `flan-t5-base` (250M params) cannot do reading comprehension or reproduce tables. Groq's `llama-3.1-8b-instant` is free and handles structured output correctly |
| **SSE over WebSockets for streaming** | SSE is unidirectional (server→client), simpler, reconnects automatically, and sufficient for this use case |

---

## 7. API Endpoints

| Method | Endpoint | Parameters | Response | Description |
|---|---|---|---|---|
| GET | `/` | — | HTML | Serves `interface/index.html` |
| GET | `/workflow` | — | HTML | Serves `interface/workflow.html` |
| GET | `/retrieve` | `prompt` (string) | JSON array | Returns top-8 chunk dicts with all scores |
| GET | `/generate` | `prompt` (string) | SSE stream | Full retrieve+generate pipeline, streamed |

**SSE Events emitted by `/generate`:**

| Event | Data | When |
|---|---|---|
| `start` | `"starting"` | Immediately on connection |
| `delta` | token string | Each answer token from Groq |
| `replace` | full answer string | When Groq answer is invalid → extractive fallback |
| `sources` | JSON array of chunks | After generation completes |
| `done` | `"finished"` | Stream end |
| `error` | JSON `{message}` | On exception |

---

## 8. Dependencies

**Python version:** 3.10–3.12 (PyTorch 2.x does not support Python 3.13)

```
pip install -r requirements.txt
```

**Quick start:**
```bash
# Step 1 — Build index (run once)
python build_index.py

# Step 2 — Start server
python server.py

# Open browser at http://localhost:8000

# Optional — run experiments
python experiments.py
```

**Groq API key (free):**
1. Sign up at https://console.groq.com/
2. Create an API key (starts with `gsk_...`)
3. Add to `.env`: `GROQ_API_KEY=gsk_your_key_here`
