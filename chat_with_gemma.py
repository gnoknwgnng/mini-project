from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For dev only; restrict in production!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "gemma3:latest"

class PromptRequest(BaseModel):
    prompt: str

@app.post("/generate")
def generate_text(req: PromptRequest):
    payload = {
        "model": MODEL_NAME,
        "prompt": req.prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    return {"response": data.get("response", "")}