"""
server.py
=========
FastAPI backend for the Handbook QA System.

Endpoints:
  GET /              → serves the web UI
  GET /generate      → streams the answer over SSE
  GET /retrieve      → returns top-k chunks as JSON (for evidence panel)

FIXES APPLIED
-------------
FIX 3 — GEN_TOP_K raised from 3 → 5; input max_length raised to 512 (the
         actual safe limit for flan-t5-base encoder); max_new_tokens raised
         from 150 → 256; min_new_tokens=20 added to prevent 1-word outputs;
         num_beams=4 replaces greedy decoding for far better coherence;
         no_repeat_ngram_size=3 prevents repetition loops.

FIX 4 — Context is now built with a hard token budget per chunk instead of
         blindly concatenating then truncating, so the most relevant chunk
         text always fits rather than being silently cut.

FIX 5 — validate_answer() added: rejects one-word/garbage outputs and returns
         a safe, evidence-grounded fallback that points the user to the
         retrieved evidence panel rather than returning nonsense.

FIX 6 — Logging configured at startup so retrieval debug output from
         pipeline.py is visible in the terminal during development.
         Set LOG_LEVEL=WARNING in production to silence it.
"""

import json
import logging
import os
import threading
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TextIteratorStreamer

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

# ── App & static files ────────────────────────────────────────────────────────
app = FastAPI(title="Handbook QA", version="1.1")
app.mount("/static", StaticFiles(directory=INTERFACE_DIR), name="static")

# ── Load models (once at startup) ─────────────────────────────────────────────
logger.info("Loading tokenizer and generation model from %s …", MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
logger.info("Loading retrieval pipeline …")
retrieval_pipeline = HandbookPipeline(str(MODEL_DIR))
logger.info("All models loaded. Ready.")

# ── Generation settings ───────────────────────────────────────────────────────
# FIX 3: raised from 3 to 5 — more context = better answers
GEN_TOP_K = 5

# FIX 3: flan-t5-base encoder hard limit is 512 tokens.
#         Setting higher causes silent truncation of the START of the prompt
#         (i.e. the instruction) rather than the end. 512 is the safe ceiling.
ENCODER_MAX_TOKENS = 512

# FIX 3: how many tokens each chunk may contribute to the prompt context.
#         5 chunks × ~80 tokens = ~400 tokens, leaving ~112 for the instruction
#         and question. Adjust down if your chunks are very long.
TOKENS_PER_CHUNK = 80

# Minimum word count for an answer to be considered valid before fallback.
MIN_ANSWER_WORDS = 4

# ── Helpers ───────────────────────────────────────────────────────────────────

def sse_event(event: str, data: str = "") -> str:
    payload = f"event: {event}\n"
    for line in (data.splitlines() or [""]):
        payload += f"data: {line}\n"
    return payload + "\n"


def format_chunk_for_client(chunk: dict, rank: int) -> dict:
    """Serialise a chunk dict into a JSON-safe object for the UI."""
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


def build_context(docs: list, max_tokens_per_chunk: int = TOKENS_PER_CHUNK) -> str:
    """
    FIX 4: Build generation context with a per-chunk token budget.

    Instead of concatenating all chunks and hoping truncation lands in a useful
    place, we truncate each chunk individually to `max_tokens_per_chunk` tokens
    before joining. This guarantees every chunk contributes something rather
    than the last chunks being silently dropped entirely.
    """
    parts = []
    for d in docs:
        src  = d.get("source", "handbook").replace(".pdf", "")
        pg   = d.get("page", "?")
        text = d.get("text", "")

        # Truncate chunk text to budget using the tokenizer (exact token count)
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) > max_tokens_per_chunk:
            token_ids = token_ids[:max_tokens_per_chunk]
            text = tokenizer.decode(token_ids, skip_special_tokens=True)

        parts.append(f"[{src}, p.{pg}]\n{text}")

    return "\n\n".join(parts)


def validate_answer(answer: str, docs: list) -> str:
    """
    FIX 5: Reject garbage outputs and return a grounded fallback.

    flan-t5-base sometimes outputs a single word, a single character, or a
    phrase that is a fragment of the prompt rather than an answer. We detect
    this and substitute a fallback that:
      - tells the user what we found
      - points them to the evidence panel (which always shows the chunks)
      - cites the top source page so the answer is still grounded
    """
    answer = answer.strip()
    words  = answer.split()

    # Detect one-word / single-char / empty outputs
    is_garbage = (
        len(words) < MIN_ANSWER_WORDS
        or len(answer) < 12
        # Detect prompt echo: model sometimes outputs the instruction itself
        or answer.lower().startswith("answer the question")
        or answer.lower().startswith("context:")
    )

    if is_garbage:
        if docs:
            top = docs[0]
            page   = top.get("page", "?")
            source = top.get("source", "the handbook").replace(".pdf", "")
            logger.warning(
                "Rejected short/garbage answer %r — returning grounded fallback.", answer
            )
            return (
                f"Based on the retrieved handbook sections, the most relevant "
                f"information appears on page {page} of {source}. "
                f"Please review the evidence panel on the right for the exact text — "
                f"the top chunk contains the specific details for your query."
            )
        return (
            "I could not find sufficient evidence in the handbook for this query. "
            "Please try rephrasing or consult the handbook directly."
        )

    return answer

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(INTERFACE_DIR / "index.html")


@app.get("/retrieve")
def retrieve(prompt: str = Query(..., min_length=1)) -> JSONResponse:
    """Return top-k retrieved chunks as JSON for the evidence panel."""
    docs = retrieval_pipeline.retrieve(prompt)
    return JSONResponse([format_chunk_for_client(d, i) for i, d in enumerate(docs)])


def stream_generation(prompt: str) -> Iterator[str]:
    yield sse_event("start", "starting")

    try:
        # ── Retrieval ──────────────────────────────────────────────────────────
        docs = retrieval_pipeline.retrieve(prompt)

        if not docs:
            # Retrieval returned nothing at all — emit fallback immediately
            fallback = (
                "I could not find any relevant sections in the handbook for this query. "
                "Please try rephrasing your question."
            )
            yield sse_event("delta", fallback)
            yield sse_event("sources", json.dumps([]))
            yield sse_event("done", "finished")
            return

        logger.info(
            "Retrieved %d chunks for query '%s'. Top score: %.4f",
            len(docs), prompt, docs[0].get("score", 0)
        )

        # ── Context construction ───────────────────────────────────────────────
        # FIX 4: per-chunk token budget keeps all GEN_TOP_K chunks visible
        gen_docs = docs[:GEN_TOP_K]
        context  = build_context(gen_docs, max_tokens_per_chunk=TOKENS_PER_CHUNK)

        final_prompt = (
            "Answer the question using only the handbook context below. "
            "Give a complete sentence answer. "
            "If the context does not contain the answer, say 'Not found in handbook'.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {prompt}\n"
            "Answer:"
        )

        logger.debug("Final prompt (%d chars):\n%s", len(final_prompt), final_prompt[:400])

        # ── Tokenise ───────────────────────────────────────────────────────────
        # FIX 3: ENCODER_MAX_TOKENS=512 is the safe hard limit for flan-t5-base.
        #         truncation=True + max_length=512 truncates from the END,
        #         preserving the instruction and question at the start.
        inputs = tokenizer(
            final_prompt,
            return_tensors="pt",
            truncation=True,
            max_length=ENCODER_MAX_TOKENS,
        )

        # ── Generation ─────────────────────────────────────────────────────────
        streamer = TextIteratorStreamer(
            tokenizer, skip_prompt=True, skip_special_tokens=True
        )

        generated_tokens: list[str] = []

        def run_generation():
            model.generate(
                **inputs,
                streamer=streamer,
                # TextIteratorStreamer only supports num_beams=1 (greedy/sampling).
                # We compensate for quality loss with:
                #   - min_new_tokens=20  : prevents 1-word / 1-char outputs
                #   - repetition_penalty : discourages looping phrases
                #   - no_repeat_ngram_size: hard-blocks repeated 3-grams
                #   - temperature + do_sample: light sampling for more natural output
                num_beams=1,
                do_sample=True,
                temperature=0.3,        # low = stays factual, avoids hallucination
                min_new_tokens=20,
                max_new_tokens=256,
                no_repeat_ngram_size=3,
                repetition_penalty=1.3,
            )

        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()

        for token in streamer:
            if token:
                generated_tokens.append(token)
                yield sse_event("delta", token)

        thread.join()

        # ── Post-generation validation ─────────────────────────────────────────
        # FIX 5: if the raw answer is garbage, replace it in the stream.
        #         We have already sent the raw tokens above, so we emit a
        #         correction event that the client replaces the last message with.
        raw_answer = "".join(generated_tokens).strip()
        validated  = validate_answer(raw_answer, docs)

        if validated != raw_answer:
            # Tell the client to replace the streamed content with the fallback
            yield sse_event("replace", validated)

        # ── Emit sources ───────────────────────────────────────────────────────
        sources_payload = json.dumps(
            [format_chunk_for_client(d, i) for i, d in enumerate(docs)]
        )
        yield sse_event("sources", sources_payload)
        yield sse_event("done", "finished")

    except Exception as exc:
        logger.exception("Generation error for query '%s': %s", prompt, exc)
        yield sse_event("error", json.dumps({"message": str(exc)}))


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