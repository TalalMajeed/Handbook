from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

app = FastAPI()

interface_dir = Path(__file__).resolve().parent / "interface"
app.mount("/", StaticFiles(directory=interface_dir, html=True), name="interface")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)