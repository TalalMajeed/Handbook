from pathlib import Path
import json
import threading
from typing import Iterator

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TextIteratorStreamer

from pipeline import HandbookPipeline


BASE_DIR = Path(__file__).resolve().parent
INTERFACE_DIR = BASE_DIR / "interface"
MODEL_DIR = BASE_DIR / "handbook-1.0"


app = FastAPI()

app.mount("/static", StaticFiles(directory=INTERFACE_DIR), name="static")


tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR)

pipeline = HandbookPipeline(str(MODEL_DIR))


@app.get("/")
def index() -> FileResponse:
    return FileResponse(INTERFACE_DIR / "index.html")


def sse_event(event: str, data: str = "") -> str:
    payload = f"event: {event}\n"
    if data:
        for line in data.splitlines() or [""]:
            payload += f"data: {line}\n"
    else:
        payload += "data: \n"
    return payload + "\n"


def stream_generation(prompt: str) -> Iterator[str]:
    yield sse_event("start", "starting")

    try:
        docs = pipeline.retrieve(prompt)

        context = "\n\n".join(docs)

        if not context.strip():
            final_prompt = f"Question: {prompt}\nAnswer:"
        else:
            final_prompt = f"""
Answer strictly from context.

Context:
{context}

Question: {prompt}
Answer:
"""

        inputs = tokenizer(final_prompt, return_tensors="pt")

        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True
        )

        def run_generation():
            model.generate(
                **inputs,
                streamer=streamer,
                max_new_tokens=128,
                do_sample=True,
                temperature=0.7,
            )

        thread = threading.Thread(target=run_generation, daemon=True)
        thread.start()

        for chunk in streamer:
            if chunk:
                yield sse_event("delta", chunk)

        thread.join()

        yield sse_event("done", "finished")

    except Exception as exc:
        yield sse_event("error", json.dumps({"message": str(exc)}))


@app.get("/generate")
def generate(prompt: str = Query(..., min_length=1)) -> StreamingResponse:
    return StreamingResponse(
        stream_generation(prompt),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)