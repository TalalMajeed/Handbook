from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import pipeline

class Input(BaseModel):
    prompt: str

app = FastAPI()

interface_dir = Path(__file__).resolve().parent / "interface"
app.mount("/", StaticFiles(directory=interface_dir, html=True), name="interface")

pipe = pipeline(
    "text-generation",
    model="handbook-1.0"
)

@app.post("/generate")
def generate(data: Input):
    result = pipe(data.prompt, max_length=128)
    return {"response": result[0]["generated_text"]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)