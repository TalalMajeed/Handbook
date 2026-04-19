"""
build_index.py
==============
Indexing pipeline for the Handbook QA System.

Loads both handbooks (UG + PG), chunks them, deduplicates with real SimHash,
computes sentence embeddings, builds a PageRank graph, fits TF-IDF, and
constructs a MinHash + LSH index. All artifacts are saved to OUTPUT_DIR.

Run:
    python build_index.py
"""

import os
import re
import pickle
import hashlib
import time

import fitz                     # PyMuPDF
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

from lsh_core import MinHasher, LSHIndex, N_HASHES, N_BANDS, SHINGLE_SIZE

# ── Configuration ─────────────────────────────────────────────────────────────

PDF_PATHS = [
    "./data/ug-handbook.pdf",   # UG Handbook
    "./data/pg-handbook.pdf",   # PG Handbook
]

CHUNK_MIN_WORDS  = 60
CHUNK_MAX_WORDS  = 300
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR       = "handbook-1.0"

# SimHash
SIMHASH_BITS      = 64
SIMHASH_THRESHOLD = 3          # max Hamming distance to call two chunks duplicates

# MinHash + LSH (constants imported from lsh_core)
# N_HASHES = 128, N_BANDS = 16, SHINGLE_SIZE = 3

# PageRank
SIMILARITY_THRESHOLD = 0.65
PAGERANK_DAMPING     = 0.85

# TF-IDF
TFIDF_MAX_FEATURES = 20_000

# ── 1. Data Ingestion ─────────────────────────────────────────────────────────

def load_pdf(path):
    """Extract per-page text from a PDF."""
    doc = fitz.open(path)
    pages = []
    source = os.path.basename(path)
    for i, page in enumerate(doc):
        text = re.sub(r"\s+", " ", page.get_text()).strip()
        if text:
            pages.append({"page": i + 1, "source": source, "text": text})
    print(f"  Loaded {len(pages)} pages from {source}")
    return pages


def chunk_pages(pages):
    """
    Split page text into overlapping word-window chunks.
    Each chunk carries page number and source file.
    """
    chunks = []
    for p in pages:
        words = p["text"].split()
        step  = CHUNK_MAX_WORDS // 2          # 50 % overlap
        start = 0
        while start < len(words):
            end   = min(start + CHUNK_MAX_WORDS, len(words))
            chunk_words = words[start:end]
            if len(chunk_words) >= CHUNK_MIN_WORDS:
                chunks.append({
                    "page":   p["page"],
                    "source": p["source"],
                    "text":   " ".join(chunk_words),
                })
            if end == len(words):
                break
            start += step
    print(f"  Created {len(chunks)} raw chunks")
    return chunks

# ── 2A. Real SimHash ──────────────────────────────────────────────────────────

def compute_simhash(text, n_bits=SIMHASH_BITS):
    """
    SimHash fingerprint of `text`.

    Algorithm:
      1. Tokenise into words.
      2. For each token, compute an MD5 hash and project each bit (+1 / -1)
         onto an n_bits accumulator vector.
      3. Convert the vector to a fingerprint by taking the sign of each position.
    """
    tokens = text.lower().split()
    v = np.zeros(n_bits, dtype=np.float64)
    for token in tokens:
        h = int(hashlib.md5(token.encode()).hexdigest(), 16)
        for i in range(n_bits):
            v[i] += 1 if (h >> i) & 1 else -1
    fp = 0
    for i in range(n_bits):
        if v[i] > 0:
            fp |= 1 << i
    return fp


def hamming_distance(h1, h2):
    """Number of differing bits between two integer fingerprints."""
    return bin(h1 ^ h2).count("1")


def simhash_dedup(chunks):
    """Remove near-duplicate chunks using SimHash + Hamming distance."""
    fps   = [compute_simhash(c["text"]) for c in chunks]
    keep  = []
    kept_fps = []
    for i, c in enumerate(chunks):
        is_dup = any(hamming_distance(fps[i], kfp) <= SIMHASH_THRESHOLD
                     for kfp in kept_fps)
        if not is_dup:
            c["simhash"] = fps[i]
            keep.append(c)
            kept_fps.append(fps[i])
    print(f"  After SimHash dedup: {len(keep)} chunks "
          f"(removed {len(chunks) - len(keep)})")
    return keep

# ── 2B. MinHash + LSH ─────────────────────────────────────────────────────────
# MinHasher and LSHIndex are imported from lsh_core.py

# ── 3. Embeddings ─────────────────────────────────────────────────────────────

def compute_embeddings(chunks, embed_model):
    texts = [c["text"] for c in chunks]
    embeddings = embed_model.encode(texts, normalize_embeddings=True,
                                    show_progress_bar=True, batch_size=64)
    for i, c in enumerate(chunks):
        c["embedding"] = embeddings[i]
    return embeddings

# ── 4. PageRank ───────────────────────────────────────────────────────────────

def compute_pagerank(chunks, embeddings):
    G = nx.Graph()
    n = len(chunks)
    G.add_nodes_from(range(n))
    edges_added = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim > SIMILARITY_THRESHOLD:
                G.add_edge(i, j, weight=sim)
                edges_added += 1
    print(f"  PageRank graph: {n} nodes, {edges_added} edges")
    pr = nx.pagerank(G, alpha=PAGERANK_DAMPING)
    for i, c in enumerate(chunks):
        c["pagerank"] = pr.get(i, 0.0)

# ── 5. TF-IDF ─────────────────────────────────────────────────────────────────

def build_tfidf(chunks):
    texts     = [c["text"] for c in chunks]
    vectorizer = TfidfVectorizer(max_features=TFIDF_MAX_FEATURES)
    matrix    = vectorizer.fit_transform(texts)
    print(f"  TF-IDF matrix: {matrix.shape}")
    return vectorizer, matrix

# ── 6. Save Artifacts ─────────────────────────────────────────────────────────

def save_artifacts(chunks, vectorizer, tfidf_matrix, embed_model,
                   minhash_sigs, lsh_index, minhaser):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(f"{OUTPUT_DIR}/chunks.pkl", "wb") as f:
        pickle.dump(chunks, f)

    with open(f"{OUTPUT_DIR}/tfidf.pkl", "wb") as f:
        pickle.dump((vectorizer, tfidf_matrix), f)

    with open(f"{OUTPUT_DIR}/lsh.pkl", "wb") as f:
        pickle.dump((minhaser, minhash_sigs, lsh_index), f)

    embed_model.save(f"{OUTPUT_DIR}/embed_model")

    print(f"\n  All artifacts saved to ./{OUTPUT_DIR}/")
    print(f"    chunks.pkl      -- {len(chunks)} chunks")
    print(f"    tfidf.pkl       -- TF-IDF vectorizer + matrix")
    print(f"    lsh.pkl         -- MinHasher + signatures + LSHIndex")
    print(f"    embed_model/    -- sentence-transformers model")

GEN_MODEL_NAME = "google/flan-t5-base"

# ── 7. Save Generation Model ──────────────────────────────────────────────────

def save_gen_model():
    """Download google/flan-t5-base and save to OUTPUT_DIR (skip if exists)."""
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    marker = os.path.join(OUTPUT_DIR, "config.json")
    if os.path.exists(marker):
        print(f"  flan-t5-base already in ./{OUTPUT_DIR}/, skipping download.")
        return
    print(f"  Downloading {GEN_MODEL_NAME} ...")
    tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    tok.save_pretrained(OUTPUT_DIR)
    mdl.save_pretrained(OUTPUT_DIR)
    print(f"  Generation model saved to ./{OUTPUT_DIR}/")

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("\n[1/7] Loading PDFs ...")
    all_pages = []
    for path in PDF_PATHS:
        if os.path.exists(path):
            all_pages.extend(load_pdf(path))
        else:
            print(f"  WARNING: {path} not found, skipping.")
    print(f"  Total pages across all sources: {len(all_pages)}")

    print("\n[2/7] Chunking ...")
    chunks = chunk_pages(all_pages)

    print("\n[3/7] SimHash deduplication ...")
    chunks = simhash_dedup(chunks)

    print("\n[4/7] Computing sentence embeddings ...")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings  = compute_embeddings(chunks, embed_model)

    print("\n[5/7] Building PageRank graph ...")
    compute_pagerank(chunks, embeddings)

    print("\n[6/7] Building TF-IDF index and MinHash+LSH index ...")
    vectorizer, tfidf_matrix = build_tfidf(chunks)

    minhaser     = MinHasher(n_hashes=N_HASHES)
    lsh_index    = LSHIndex(n_hashes=N_HASHES, n_bands=N_BANDS)
    minhash_sigs = []
    print(f"  LSH threshold s* ~= {lsh_index.threshold:.3f} "
          f"(n_bands={N_BANDS}, rows_per_band={lsh_index.rows_per_band})")
    for i, c in enumerate(chunks):
        sig = minhaser.signature(c["text"])
        c["minhash"] = sig
        minhash_sigs.append(sig)
        lsh_index.add(i, sig)
    print(f"  MinHash+LSH index built for {len(chunks)} chunks")

    print("\n[Saving retrieval artifacts] ...")
    save_artifacts(chunks, vectorizer, tfidf_matrix, embed_model,
                   minhash_sigs, lsh_index, minhaser)

    print("\n[7/7] Saving generation model ...")
    save_gen_model()

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
