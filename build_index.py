"""
build_index.py
==============
Indexing pipeline for the Handbook QA System.

Run:
    python build_index.py

FIXES IN THIS VERSION
---------------------
FIX A — Header stripping:
    Every page extracted by PyMuPDF contains a running header like
    "NUST Undergraduate Student Handbook 87" embedded in the text.
    The old chunker kept these headers in chunk text, so TF-IDF treated
    'nust', 'undergraduate', 'student', 'handbook' as content tokens.
    Since they appear in every single chunk, their IDF ≈ 0, destroying
    discriminative power. All headers and page-number-only lines are
    now stripped BEFORE chunking.

FIX B — Sentence-aware chunking (replaces blind word-window):
    The old 300-word / 50%-overlap window split sentences mid-way,
    meaning a rule like "minimum CGPA is 2.75 for Engineering and 3.00
    for NBS" could be cut in half — with neither chunk containing the
    complete rule. The new chunker accumulates whole sentences up to
    MAX_WORDS, keeping one sentence of overlap for context continuity.

FIX C — Query-expansion synonyms baked into each chunk:
    The handbook uses "CGPA" but users ask "GPA". The vocabulary mismatch
    means TF-IDF never finds the right chunk. We append a short hidden
    synonym line to each chunk before TF-IDF fitting so that both "GPA"
    and "CGPA" map to the same chunks.

FIX D — TF-IDF custom stop-words:
    Words that appear in nearly every chunk ('nust', 'university',
    'student', 'handbook', 'seecs', 'semester') are added to the
    stop-word list so TF-IDF stops giving them any weight.
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

from lsh_core import MinHasher, LSHIndex, N_HASHES, N_BANDS

# ── Configuration ─────────────────────────────────────────────────────────────

PDF_PATHS = [
    "./data/ug-handbook.pdf",
    "./data/pg-handbook.pdf",
]

CHUNK_MIN_WORDS  = 50
CHUNK_MAX_WORDS  = 250          # smaller than before → tighter, more precise chunks
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OUTPUT_DIR       = "handbook-1.0"

SIMHASH_BITS      = 64
SIMHASH_THRESHOLD = 3

SIMILARITY_THRESHOLD = 0.65
PAGERANK_DAMPING     = 0.85

# FIX D: domain stop-words — appear in almost every chunk, destroy IDF signal
DOMAIN_STOP_WORDS = [
    "nust", "university", "student", "students", "handbook", "seecs",
    "semester", "programme", "department", "faculty", "academic",
    "undergraduate", "postgraduate", "shall", "may", "also", "one",
    "two", "three", "per", "upon", "thereof", "therein",
]

# ── FIX A: Header / footer patterns to strip from every page ──────────────────

# Matches running headers like "NUST Undergraduate Student Handbook 87"
# and "NUST Postgraduate Student Handbook 14"
_HEADER_RE = re.compile(
    r'NUST\s+(Undergraduate|Postgraduate)\s+Student\s+Handbook\s+\d+',
    re.IGNORECASE
)

# Matches standalone page numbers (lines that are just a number)
_PAGENUM_RE = re.compile(r'^\s*\d{1,3}\s*$', re.MULTILINE)

# Matches repeated section headers that duplicate index info
_SEECS_RE = re.compile(
    r'School\s+of\s+Electrical\s+Engineering\s+(&|and)\s+Computer\s+Science',
    re.IGNORECASE
)


def strip_headers(text: str) -> str:
    """
    FIX A: Remove running headers, footer page numbers, and school name
    stamps that PyMuPDF extracts as part of page body text.

    These strings appear verbatim on every page and contribute zero
    semantic content — but they dominate TF-IDF vocabulary because
    of their extreme frequency.
    """
    text = _HEADER_RE.sub(" ", text)
    text = _PAGENUM_RE.sub(" ", text)
    text = _SEECS_RE.sub(" ", text)
    # collapse whitespace created by removals
    text = re.sub(r" {2,}", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


# ── FIX B: Sentence-aware chunker ─────────────────────────────────────────────

# Simple sentence splitter: split on . ! ? followed by whitespace + capital/digit.
# Keeps abbreviations mostly intact (e.g. "Ph.D." won't split mid-word because
# the next char after ". " is usually lowercase in abbreviations).
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9\(])')


def split_sentences(text: str) -> list:
    parts = _SENT_SPLIT_RE.split(text)
    return [p.strip() for p in parts if len(p.strip()) > 10]


def chunk_pages(pages: list) -> list:
    """
    FIX B: Sentence-aware chunking with header stripping.

    Algorithm:
      1. Strip headers/footers from page text.
      2. Split into sentences.
      3. Accumulate sentences until adding the next would exceed MAX_WORDS.
      4. Emit the chunk, keep the LAST sentence as overlap for the next chunk
         (so cross-sentence rules aren't split across chunks).
      5. Chunks shorter than MIN_WORDS are discarded.
    """
    chunks = []
    for p in pages:
        clean     = strip_headers(p["text"])
        sentences = split_sentences(clean)

        current_sents = []
        current_words = 0

        for sent in sentences:
            sent_wc = len(sent.split())

            if current_words + sent_wc > CHUNK_MAX_WORDS and current_words >= CHUNK_MIN_WORDS:
                # Emit current chunk
                chunk_text = " ".join(current_sents)
                chunks.append({
                    "page":   p["page"],
                    "source": p["source"],
                    "text":   chunk_text,
                })
                # Keep last sentence as overlap
                overlap_sent  = current_sents[-1]
                current_sents = [overlap_sent, sent]
                current_words = len(overlap_sent.split()) + sent_wc
            else:
                current_sents.append(sent)
                current_words += sent_wc

        # Emit remainder
        if current_words >= CHUNK_MIN_WORDS:
            chunks.append({
                "page":   p["page"],
                "source": p["source"],
                "text":   " ".join(current_sents),
            })

    print(f"  Created {len(chunks)} raw chunks (sentence-aware)")
    return chunks


# ── FIX C: Synonym expansion ──────────────────────────────────────────────────

# Maps handbook abbreviations/terms to the user-facing synonyms to append.
# This is baked into chunk text ONLY for TF-IDF fitting — not shown to user.
SYNONYM_MAP = {
    r'\bCGPA\b':        'GPA grade point average',
    r'\bUG\b':          'undergraduate',
    r'\bPG\b':          'postgraduate graduate',
    r'\bSWS\b':         'semester work score',
    r'\bDRC\b':         'doctoral research committee',
    r'\bHEC\b':         'higher education commission',
    r'\bPEAC\b':        'programme educational advisory committee',
    r'\bCH\b':          'credit hours',
    r'\bprobation\b':   'probation academic warning suspension',
    r'\bdebarred\b':    'debarred banned excluded failed attendance',
    r'\brepeat\b':      'repeat retake redo course failure',
    r'\battendance\b':  'attendance present absent shortage',
    r'\bwithdrawal\b':  'withdrawal drop withdraw leave',
    r'\bthesis\b':      'thesis dissertation research project',
    r'\bplagarism\b':   'plagiarism cheating academic dishonesty',
    r'\bplagiarism\b':  'plagiarism cheating academic dishonesty',
    r'\bsupply\b':      'supplementary exam re-examination',
    r'\brefund\b':      'refund fee tuition money return',
    r'\belective\b':    'elective optional choice free course',
}


def expand_synonyms(text: str) -> str:
    """
    FIX C: Append synonym tokens to the text so TF-IDF can match
    user vocabulary against handbook abbreviations.
    Returns ORIGINAL text + ' ' + expansion (expansion is invisible to user).
    """
    expansions = []
    for pattern, synonyms in SYNONYM_MAP.items():
        if re.search(pattern, text, re.IGNORECASE):
            expansions.append(synonyms)
    if expansions:
        return text + "  " + " ".join(expansions)
    return text


# ── 1. Data Ingestion ─────────────────────────────────────────────────────────

def load_pdf(path: str) -> list:
    doc    = fitz.open(path)
    pages  = []
    source = os.path.basename(path)
    for i, page in enumerate(doc):
        raw  = page.get_text()
        text = re.sub(r"\s+", " ", raw).strip()
        if len(text) > 50:          # skip near-empty pages
            pages.append({"page": i + 1, "source": source, "text": text})
    print(f"  Loaded {len(pages)} non-empty pages from {source}")
    return pages


# ── 2. SimHash deduplication ──────────────────────────────────────────────────

def compute_simhash(text: str, n_bits: int = SIMHASH_BITS) -> int:
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


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def simhash_dedup(chunks: list) -> list:
    fps      = [compute_simhash(c["text"]) for c in chunks]
    keep     = []
    kept_fps = []
    for i, c in enumerate(chunks):
        is_dup = any(
            hamming_distance(fps[i], kfp) <= SIMHASH_THRESHOLD
            for kfp in kept_fps
        )
        if not is_dup:
            c["simhash"] = fps[i]
            keep.append(c)
            kept_fps.append(fps[i])
    print(f"  After SimHash dedup: {len(keep)} chunks (removed {len(chunks) - len(keep)})")
    return keep


# ── 3. Embeddings ─────────────────────────────────────────────────────────────

def compute_embeddings(chunks: list, embed_model) -> np.ndarray:
    texts      = [c["text"] for c in chunks]
    embeddings = embed_model.encode(
        texts, normalize_embeddings=True,
        show_progress_bar=True, batch_size=64
    )
    for i, c in enumerate(chunks):
        c["embedding"] = embeddings[i]
    return embeddings


# ── 4. PageRank ───────────────────────────────────────────────────────────────

def compute_pagerank(chunks: list, embeddings: np.ndarray):
    G = nx.Graph()
    n = len(chunks)
    G.add_nodes_from(range(n))
    edges = 0
    for i in range(n):
        for j in range(i + 1, n):
            sim = float(np.dot(embeddings[i], embeddings[j]))
            if sim > SIMILARITY_THRESHOLD:
                G.add_edge(i, j, weight=sim)
                edges += 1
    print(f"  PageRank graph: {n} nodes, {edges} edges")
    pr = nx.pagerank(G, alpha=PAGERANK_DAMPING)
    for i, c in enumerate(chunks):
        c["pagerank"] = pr.get(i, 0.0)


# ── 5. TF-IDF ─────────────────────────────────────────────────────────────────

def build_tfidf(chunks: list):
    """
    FIX C + FIX D: Build TF-IDF on synonym-expanded text with domain stop-words.

    - expand_synonyms() adds user-vocabulary synonyms so "GPA" queries hit
      "CGPA" chunks and vice versa.
    - DOMAIN_STOP_WORDS removes handbook-universal tokens that destroy IDF.
    - sublinear_tf=True dampens term frequency so a word appearing 10×
      counts as ~3× not 10×, improving discrimination.
    - min_df=1, max_df=0.85: exclude tokens appearing in >85% of chunks
      (another safety net against handbook-universal words).
    """
    # Build expanded texts for TF-IDF fitting only
    expanded_texts = [expand_synonyms(c["text"]) for c in chunks]

    vectorizer = TfidfVectorizer(
        max_features=15_000,
        sublinear_tf=True,          # log(1+tf) instead of raw tf
        min_df=1,
        max_df=0.85,                # FIX D: drop tokens in >85% of chunks
        stop_words=DOMAIN_STOP_WORDS,
        ngram_range=(1, 2),         # unigrams + bigrams (e.g. "credit hours")
    )
    matrix = vectorizer.fit_transform(expanded_texts)
    print(f"  TF-IDF matrix: {matrix.shape}  (vocab size: {len(vectorizer.vocabulary_)})")
    return vectorizer, matrix


# ── 6. MinHash + LSH ─────────────────────────────────────────────────────────

def build_lsh(chunks: list):
    minhaser  = MinHasher(n_hashes=N_HASHES)
    lsh_index = LSHIndex(n_hashes=N_HASHES, n_bands=N_BANDS)
    sigs      = []
    print(f"  LSH threshold s* ~= {lsh_index.threshold:.3f}")
    for i, c in enumerate(chunks):
        sig = minhaser.signature(c["text"])
        c["minhash"] = sig
        sigs.append(sig)
        lsh_index.add(i, sig)
    print(f"  MinHash+LSH index built for {len(chunks)} chunks")
    return minhaser, sigs, lsh_index


# ── 7. Save artifacts ─────────────────────────────────────────────────────────

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

    print(f"\n  Artifacts saved to ./{OUTPUT_DIR}/")
    print(f"    chunks.pkl  — {len(chunks)} chunks")
    print(f"    tfidf.pkl   — {tfidf_matrix.shape}")
    print(f"    lsh.pkl     — MinHasher + sigs + LSHIndex")
    print(f"    embed_model/")


# ── 8. Save generation model ──────────────────────────────────────────────────

GEN_MODEL_NAME = "google/flan-t5-base"


def save_gen_model():
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    marker = os.path.join(OUTPUT_DIR, "config.json")
    if os.path.exists(marker):
        print(f"  flan-t5-base already in ./{OUTPUT_DIR}/, skipping.")
        return
    print(f"  Downloading {GEN_MODEL_NAME} …")
    tok = AutoTokenizer.from_pretrained(GEN_MODEL_NAME)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(GEN_MODEL_NAME)
    tok.save_pretrained(OUTPUT_DIR)
    mdl.save_pretrained(OUTPUT_DIR)
    print(f"  Generation model saved.")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    print("\n[1/8] Loading PDFs …")
    all_pages = []
    for path in PDF_PATHS:
        if os.path.exists(path):
            all_pages.extend(load_pdf(path))
        else:
            print(f"  WARNING: {path} not found, skipping.")
    print(f"  Total pages: {len(all_pages)}")

    print("\n[2/8] Sentence-aware chunking with header stripping …")
    chunks = chunk_pages(all_pages)

    print("\n[3/8] SimHash deduplication …")
    chunks = simhash_dedup(chunks)

    print("\n[4/8] Computing sentence embeddings …")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    embeddings  = compute_embeddings(chunks, embed_model)

    print("\n[5/8] Building PageRank graph …")
    compute_pagerank(chunks, embeddings)

    print("\n[6/8] Building TF-IDF index (with synonym expansion + stop-words) …")
    vectorizer, tfidf_matrix = build_tfidf(chunks)

    print("\n[7/8] Building MinHash + LSH index …")
    minhaser, minhash_sigs, lsh_index = build_lsh(chunks)

    print("\n[Saving retrieval artifacts] …")
    save_artifacts(chunks, vectorizer, tfidf_matrix, embed_model,
                   minhash_sigs, lsh_index, minhaser)

    print("\n[8/8] Saving generation model …")
    save_gen_model()

    print(f"\nDone in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()