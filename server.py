"""
server.py
=========
FastAPI backend for the Handbook QA System.

Endpoints:
  GET /              → serves the web UI
  GET /generate      → streams the answer over SSE
  GET /retrieve      → returns top-k chunks as JSON (evidence panel)

ROOT CAUSES FIXED IN THIS VERSION
----------------------------------
RC-1  flan-t5-base cannot reproduce tables/lists reliably.
      → Replaced with Groq API (llama-3.1-8b-instant) — 100% FREE.
        The project spec explicitly allows "LLM APIs (e.g., OpenAI,
        open-source models)" for Answer Generation. ALL retrieval
        (TF-IDF, MinHash+LSH, SimHash, PageRank) stays 100% local.
        The API only reads our retrieved chunks — no PDF bypass.

RC-2  flan-t5 latches on the first span in context, ignoring the
      most relevant chunk even when it appears later in the list.
      → Context is now ordered: most relevant chunk FIRST. Claude
        reads comprehensively rather than latching on position.

RC-3  "Not found in handbook" instruction was being echoed verbatim
      by flan-t5 even when evidence existed (prompt injection bug).
      → Removed from the model prompt. Fallback is now handled in
        Python AFTER generation, never by the model itself.

RC-4  Inline source tags [src, p.N] were being copied into answers.
      → Source tags are stripped from context sent to the model.
        Citations are appended cleanly in Python after generation.

RC-5  temperature sampling + max_new_tokens caused mid-table cutoff.
      → Claude handles long structured outputs natively. Streaming
        is done via the Anthropic SDK's streaming API.

RC-6  Context token budget was too tight (80 tokens/chunk).
      → Claude supports 200k context; we send full chunk text for
        the top-5 retrieved chunks, no truncation needed.

FALLBACK BEHAVIOUR (no API key set)
-------------------------------------
If ANTHROPIC_API_KEY is not set in the environment, the server falls
back to a clean extractive method: it copies the most relevant
sentences directly from the top retrieved chunk. This is far more
reliable than flan-t5 generation and needs no extra download.

HOW TO SET UP THE API KEY
--------------------------
Option A (environment variable — recommended):
    Windows PowerShell:
        $env:ANTHROPIC_API_KEY = "sk-ant-..."
    Linux / macOS:
        export ANTHROPIC_API_KEY="sk-ant-..."
    Then: python server.py

Option B (create a .env file in the project root):
    ANTHROPIC_API_KEY=sk-ant-...
    Then: pip install python-dotenv
          Add load_dotenv() at the top of this file.

HOW TO GET A FREE GROQ API KEY (takes 2 minutes)
-------------------------------------------------
1. Go to: https://console.groq.com/
2. Sign up with Google or GitHub (free, no credit card needed)
3. Click "API Keys" in the left sidebar → "Create API Key"
4. Copy the key (starts with gsk_...)

HOW TO SET THE KEY
------------------
Windows PowerShell:
    $env:GROQ_API_KEY = "gsk_..."
    python server.py

Linux / macOS:
    export GROQ_API_KEY="gsk_..."
    python server.py

FALLBACK (no key set)
---------------------
Falls back to clean extractive mode — directly copies the most
relevant sentences from the retrieved chunks. No model needed.
"""

import json
import logging
import os
import re
import threading
from pathlib import Path

from dotenv import load_dotenv   # pip install python-dotenv
load_dotenv()                    # reads .env in the project root automatically
from typing import Iterator

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from pipeline import HandbookPipeline

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
INTERFACE_DIR = BASE_DIR / "interface"
MODEL_DIR     = BASE_DIR / "handbook-1.0"

# ── API key detection ─────────────────────────────────────────────────────────
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()
USE_GROQ     = bool(GROQ_API_KEY)

if USE_GROQ:
    try:
        from groq import Groq
        _groq = Groq(api_key=GROQ_API_KEY)
        logger.info("Groq API ready (llama-3.1-8b-instant). Generation quality: HIGH.")
    except ImportError:
        logger.warning(
            "groq package not installed. Run: pip install groq\n"
            "Falling back to extractive mode."
        )
        USE_GROQ = False
else:
    logger.info(
        "GROQ_API_KEY not set — using extractive fallback.\n"
        "  Get a FREE key at https://console.groq.com/ then:\n"
        "  Windows: $env:GROQ_API_KEY = 'gsk_...'\n"
        "  Linux:   export GROQ_API_KEY='gsk_...'"
    )

# ── Flan-T5 fallback (only loaded if Groq is NOT available) ──────────────────
_tokenizer = None
_model     = None

if not USE_GROQ:
    logger.info("Loading flan-t5-base for local generation fallback …")
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        _tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        _model     = AutoModelForSeq2SeqLM.from_pretrained(str(MODEL_DIR))
        logger.info("flan-t5-base loaded.")
    except Exception as e:
        logger.warning("flan-t5 load failed (%s). Pure extractive mode.", e)

# ── App & static files ────────────────────────────────────────────────────────
app = FastAPI(title="Handbook QA", version="2.0")
app.mount("/static", StaticFiles(directory=INTERFACE_DIR), name="static")

# ── Retrieval pipeline ────────────────────────────────────────────────────────
logger.info("Loading retrieval pipeline …")
retrieval_pipeline = HandbookPipeline(str(MODEL_DIR))
logger.info("Pipeline ready — %d chunks indexed.", len(retrieval_pipeline.chunks))

# ── Settings ──────────────────────────────────────────────────────────────────
GEN_TOP_K        = 5      # chunks sent to the generator
GROQ_MODEL       = "llama-3.1-8b-instant"   # free, fast, handles tables well
GROQ_MAX_TOKENS  = 1024                     # plenty for full grading tables etc.


# ── Helpers ───────────────────────────────────────────────────────────────────

def sse_event(event: str, data: str = "") -> str:
    payload = f"event: {event}\n"
    for line in (data.splitlines() or [""]):
        payload += f"data: {line}\n"
    return payload + "\n"


def format_chunk_for_client(chunk: dict, rank: int) -> dict:
    full_text = chunk.get("text", "")
    return {
        "rank":    rank + 1,
        "text":    full_text,
        "preview": full_text[:280],
        "page":    chunk.get("page",   "?"),
        "source":  chunk.get("source", "handbook.pdf"),
        "score":   round(chunk.get("score", 0.0), 4),
        "method":  chunk.get("method", "hybrid"),
    }


def build_context_for_model(docs: list) -> str:
    """
    RC-4 FIX: Build clean context with NO inline source tags.
    Source tags inside the context string cause flan-t5 (and sometimes
    Claude) to copy them verbatim into answers.
    We number the chunks [1], [2], … instead — clean and unambiguous.
    """
    parts = []
    for i, d in enumerate(docs):
        text = d.get("text", "").strip()
        parts.append(f"[Chunk {i+1}]\n{text}")
    return "\n\n".join(parts)


def build_citations(docs: list) -> str:
    """Build a clean citation string to append after the answer."""
    cites = []
    for i, d in enumerate(docs[:3]):
        src = d.get("source", "handbook").replace(".pdf", "")
        pg  = d.get("page", "?")
        cites.append(f"[{src}, p.{pg}]")
    return "  |  Sources: " + "  ".join(cites)


def is_answer_valid(text: str) -> bool:
    """Return True if the answer looks like real content, not garbage."""
    text = text.strip()
    if len(text.split()) < 5:
        return False
    garbage_patterns = [
        r"^not found",
        r"^i (don't|do not|cannot|can't) (know|find|have)",
        r"^the context (does not|doesn't) (contain|have|mention|provide)",
        r"^\[chunk",
        r"^answer:",
        r"^context:",
    ]
    lower = text.lower()
    for pat in garbage_patterns:
        if re.match(pat, lower):
            return False
    return True


def extractive_fallback(query: str, docs: list) -> str:
    """
    Pure extractive answer: find the most relevant sentences from
    the top chunks using simple keyword overlap. No model required.
    Returns a complete, correctly-formed answer.
    """
    if not docs:
        return "No relevant information found in the handbook for this query."

    query_words = set(re.sub(r"[^a-z0-9\s]", " ", query.lower()).split())
    # Remove stop words
    stops = {"what","is","the","a","an","of","in","for","to","and","or",
             "how","when","can","does","do","are","be","was","were","on"}
    query_words -= stops

    best_sentences = []
    seen = set()

    for doc in docs[:GEN_TOP_K]:
        text = doc.get("text", "")
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) < 6:
                continue
            sent_lower = sent.lower()
            sent_words = set(re.sub(r"[^a-z0-9\s]", " ", sent_lower).split())
            overlap    = len(query_words & sent_words)
            # Deduplicate
            key = sent_lower[:60]
            if overlap > 0 and key not in seen:
                seen.add(key)
                best_sentences.append((overlap, sent))

    if not best_sentences:
        # Last resort: return the first 3 sentences of the top chunk
        top = docs[0].get("text", "")
        sents = re.split(r'(?<=[.!?])\s+', top)
        return " ".join(sents[:3]).strip()

    # Sort by overlap, take top 4 sentences
    best_sentences.sort(key=lambda x: x[0], reverse=True)
    answer = " ".join(s for _, s in best_sentences[:4])
    return answer


# ── Groq generation ───────────────────────────────────────────────────────────

def generate_with_groq(query: str, docs: list) -> Iterator[str]:
    """
    Stream answer tokens from the Groq API (llama-3.1-8b-instant).

    Groq is FREE — no credit card, 14,400 requests/day on free tier.
    Llama-3.1-8b correctly handles tables, lists, and full policy text
    unlike flan-t5-base which truncates and garbles structured output.

    RC-2 FIX: Most relevant chunk appears as Chunk 1 (first in context).
    RC-3 FIX: No "Not found" instruction in the prompt.
    RC-4 FIX: No inline [src, p.N] tags inside the context string.
    RC-6 FIX: Full chunk text, no per-chunk token truncation.
    """
    context = build_context_for_model(docs[:GEN_TOP_K])

    system_prompt = (
        "You are a precise academic policy assistant for NUST university. "
        "Answer questions ONLY using the handbook chunks provided. "
        "Rules:\n"
        "1. Give a complete, direct answer. Never truncate mid-sentence.\n"
        "2. If the chunks contain a table or list, reproduce it fully.\n"
        "3. Do not mention chunk numbers like 'Chunk 1' in your answer.\n"
        "4. Do not add any information not present in the chunks.\n"
        "5. If the chunks genuinely do not contain the answer, say: "
        "'This information was not found in the retrieved handbook sections.'\n"
        "6. Keep answers concise: 1-5 sentences for simple facts, "
        "a complete list/table for structured data."
    )

    user_message = (
        f"Handbook sections:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer completely and accurately based only on the sections above:"
    )

    # Groq supports streaming with the same interface as OpenAI
    stream = _groq.chat.completions.create(
        model=GROQ_MODEL,
        max_tokens=GROQ_MAX_TOKENS,
        temperature=0.1,        # near-zero = factual, no hallucination
        stream=True,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


# ── Flan-T5 local generation (fallback) ───────────────────────────────────────

def generate_with_flanT5(query: str, docs: list) -> str:
    """
    Local flan-t5-base generation with all prompt bugs fixed.
    Used only when Claude API is not available.

    RC-3 FIX: No "Not found" in prompt.
    RC-4 FIX: No inline source tags.
    RC-5 FIX: do_sample=False for deterministic output.
    """
    if _model is None or _tokenizer is None:
        return extractive_fallback(query, docs)

    # Build clean context — only chunk text, no source tags
    context_parts = []
    for d in docs[:3]:          # top-3 to stay inside 512 token limit
        text = d.get("text", "").strip()
        # Hard-limit each chunk to 120 words to fit all three
        words = text.split()
        if len(words) > 120:
            text = " ".join(words[:120])
        context_parts.append(text)
    context = "\n\n".join(context_parts)

    # RC-3 FIX: clean prompt, no fallback phrase inside it
    prompt = (
        f"Read the context and answer the question with a complete sentence.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n"
        f"Answer:"
    )

    inputs = _tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    from transformers import TextIteratorStreamer
    outputs = _model.generate(
        **inputs,
        num_beams=1,
        do_sample=False,            # RC-5: deterministic, no mid-table cutoff
        min_new_tokens=15,
        max_new_tokens=200,
        no_repeat_ngram_size=3,
        repetition_penalty=1.3,
    )
    answer = _tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    # Remove any leaked source-tag-like patterns the model might emit
    answer = re.sub(r'\[(?:ug|pg)-handbook[^\]]*\]', '', answer).strip()

    return answer if is_answer_valid(answer) else extractive_fallback(query, docs)


# ── SSE stream ────────────────────────────────────────────────────────────────

def stream_generation(prompt: str) -> Iterator[str]:
    yield sse_event("start", "starting")

    try:
        # ── Retrieval (always local) ───────────────────────────────────────────
        docs = retrieval_pipeline.retrieve(prompt)

        if not docs:
            yield sse_event("delta",
                "No relevant handbook sections were found for this query. "
                "Please try rephrasing."
            )
            yield sse_event("sources", json.dumps([]))
            yield sse_event("done", "finished")
            return

        logger.info(
            "Retrieved %d chunks | query='%s' | top_score=%.4f | top_page=%s",
            len(docs), prompt, docs[0].get("score", 0), docs[0].get("page")
        )

        # ── Generation ────────────────────────────────────────────────────────
        if USE_GROQ:
            # Stream Groq tokens directly to the client
            accumulated = []
            for token in generate_with_groq(prompt, docs):
                accumulated.append(token)
                yield sse_event("delta", token)

            raw_answer = "".join(accumulated).strip()

            # Append clean citations after the answer (RC-4)
            if is_answer_valid(raw_answer):
                citation = build_citations(docs)
                yield sse_event("delta", f"\n\n*{citation}*")
            else:
                # Model admitted it couldn't find the answer → use extractive
                extractive = extractive_fallback(prompt, docs)
                citation   = build_citations(docs)
                yield sse_event("replace", f"{extractive}\n\n*{citation}*")

        else:
            # Flan-T5 or pure extractive — generate synchronously then stream
            import time

            if _model is not None:
                answer = generate_with_flanT5(prompt, docs)
            else:
                answer = extractive_fallback(prompt, docs)

            citation = build_citations(docs)
            full     = f"{answer}\n\n*{citation}*"

            # Stream word-by-word so the UI doesn't freeze on long answers
            for word in full.split(" "):
                yield sse_event("delta", word + " ")
                time.sleep(0.012)   # ~80 words/sec — fast enough to feel live

        # ── Emit sources ──────────────────────────────────────────────────────
        sources_payload = json.dumps(
            [format_chunk_for_client(d, i) for i, d in enumerate(docs)]
        )
        yield sse_event("sources", sources_payload)
        yield sse_event("done", "finished")

    except Exception as exc:
        logger.exception("Generation error for query '%s'", prompt)
        yield sse_event("error", json.dumps({"message": str(exc)}))


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(INTERFACE_DIR / "index.html")


@app.get("/workflow")
def workflow() -> FileResponse:
    return FileResponse(INTERFACE_DIR / "workflow.html")


@app.get("/retrieve")
def retrieve(prompt: str = Query(..., min_length=1)) -> JSONResponse:
    docs = retrieval_pipeline.retrieve(prompt)
    return JSONResponse([format_chunk_for_client(d, i) for i, d in enumerate(docs)])


@app.get("/generate")
def generate(prompt: str = Query(..., min_length=1)) -> StreamingResponse:
    return StreamingResponse(
        stream_generation(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control":     "no-cache",
            "Connection":        "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")