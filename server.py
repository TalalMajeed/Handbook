"""
server.py
=========
FastAPI backend for the Handbook QA System.

Endpoints:
  GET /              → serves the web UI
  GET /generate      → streams the LLM answer over SSE
  GET /retrieve      → returns top-k chunks as JSON (for UI chunk display)
"""

import json
import threading
from pathlib import Path
from typing import Iterator

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TextIteratorStreamer

from pipeline import HandbookPipeline

# ── Paths ─────────────────────────────────────────────────────────────────────

BASE_DIR      = Path(__file__).resolve().parent
INTERFACE_DIR = BASE_DIR / "interface"
MODEL_DIR     = BASE_DIR / "handbook-1.0"

# ── App & static files ────────────────────────────────────────────────────────

app = FastAPI(title="Handbook QA", version="1.0")
app.mount("/static", StaticFiles(directory=INTERFACE_DIR), name="static")

# ── Load models (once at startup) ─────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)
pipeline  = HandbookPipeline(str(MODEL_DIR))

# ── Helpers ───────────────────────────────────────────────────────────────────

def sse_event(event: str, data: str = "") -> str:
    payload = f"event: {event}\n"
    for line in (data.splitlines() or [""]):
        payload += f"data: {line}\n"
    return payload + "\n"


def format_chunk_for_client(chunk: dict, rank: int) -> dict:
    """Serialise a chunk dict into a JSON-able object for the UI."""
    full_text = chunk.get("text", "")
    return {
        "rank":    rank + 1,
        "text":    full_text,           # full text for modal
        "preview": full_text[:280],     # short preview for card
        "page":    chunk.get("page",   "?"),
        "source":  chunk.get("source", "handbook.pdf"),
        "score":   round(chunk.get("score", 0.0), 4),
        "method":  chunk.get("method", "hybrid"),
    }

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/")
def index() -> FileResponse:
    return FileResponse(INTERFACE_DIR / "index.html")


@app.get("/retrieve")
def retrieve(prompt: str = Query(..., min_length=1)) -> JSONResponse:
    """
    Return top-k retrieved chunks as JSON.
    The UI calls this to populate the source-evidence panel.
    """
    docs = pipeline.retrieve(prompt)
    return JSONResponse([format_chunk_for_client(d, i) for i, d in enumerate(docs)])


GEN_TOP_K = 3   # Only pass top-3 chunks to generation to stay within flan-t5 token limit

def stream_generation(prompt: str) -> Iterator[str]:
    yield sse_event("start", "starting")

    try:
        # Retrieve top-k chunks for the answer panel (full 8)
        docs = pipeline.retrieve(prompt)

        # Build generation context from only the top-3 most relevant chunks
        gen_docs = docs[:GEN_TOP_K]
        context_parts = []
        for d in gen_docs:
            src = d.get("source", "handbook").replace(".pdf", "")
            pg  = d.get("page", "?")
            context_parts.append(f"[{src}, p.{pg}] {d['text']}")
        context = "\n\n".join(context_parts)

        final_prompt = (
            f"Answer the question based only on the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {prompt}\n"
            f"Answer:"
        )

        inputs  = tokenizer(final_prompt, return_tensors="pt",
                            truncation=True, max_length=900)  # leave headroom for output
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True,
                                       skip_special_tokens=True)

        def run_generation():
            model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=150,
                do_sample=False
            )

        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()

        for token in streamer:
            if token:
                yield sse_event("delta", token)

        thread.join()

        # Emit source evidence after answer is streamed
        sources_payload = json.dumps(
            [format_chunk_for_client(d, i) for i, d in enumerate(docs)]
        )
        yield sse_event("sources", sources_payload)
        yield sse_event("done", "finished")

    except Exception as exc:
        yield sse_event("error", json.dumps({"message": str(exc)}))


@app.get("/generate")
def generate(prompt: str = Query(..., min_length=1)) -> StreamingResponse:
    return StreamingResponse(
        stream_generation(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "Connection":       "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)