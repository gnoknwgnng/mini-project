from fastapi import FastAPI, File, UploadFile, Response, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
import PyPDF2
from typing import List
from datetime import datetime
import os
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
from fastapi.responses import FileResponse
# Add Google Translate
try:
    from deep_translator import GoogleTranslator
except ImportError:
    GoogleTranslator = None
import whisper
from pytube import YouTube
import tempfile
from youtube_transcript_api import (
    YouTubeTranscriptApi,
    TranscriptsDisabled,
    NoTranscriptFound,
    VideoUnavailable,
    CouldNotRetrieveTranscript
)
from urllib.parse import urlparse, parse_qs

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
CHAT_LOG_FILE = "chat.txt"

class PromptRequest(BaseModel):
    prompt: str

class ChatMessage(BaseModel):
    role: str  # 'user' or 'bot'
    content: str

class ChatRequest(BaseModel):
    history: List[ChatMessage]

class ChatLogAppendRequest(BaseModel):
    role: str
    content: str

def extract_video_id(youtube_url):
    """
    Extracts the video ID from any YouTube URL (standard, shortened, with extra params).
    """
    parsed_url = urlparse(youtube_url)
    # Handle standard YouTube URLs
    if 'youtube' in parsed_url.netloc:
        query = parse_qs(parsed_url.query)
        video_id = query.get("v")
        if video_id:
            return video_id[0]
    # Handle youtu.be short URLs
    if 'youtu.be' in parsed_url.netloc:
        # The first part of the path is the video ID
        path_parts = parsed_url.path.lstrip('/').split('/')
        if path_parts and path_parts[0]:
            return path_parts[0]
    # Fallback: try pytube
    try:
        return YouTube(youtube_url).video_id
    except Exception:
        return None

@app.post("/translate")
def translate_text(req: dict):
    text = req.get("text", "")
    target_language = req.get("target_language", "en")
    # Map language name to code
    lang_map = {
        "french": "fr", "spanish": "es", "german": "de", "italian": "it", "japanese": "ja", "chinese": "zh-CN", "hindi": "hi", "arabic": "ar", "russian": "ru", "portuguese": "pt", "korean": "ko", "north korean": "ko", "english": "en"
    }
    lang_code = lang_map.get(target_language.lower(), "en")
    try:
        result = GoogleTranslator(source='auto', target=lang_code).translate(text)
        return {"response": result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/generate")
def generate_text(req: PromptRequest):
    # Detect translation prompt
    import re
    match = re.match(r"translate the following text into ([^:]+):(.+)", req.prompt.strip(), re.IGNORECASE | re.DOTALL)
    if match:
        target_language = match.group(1).strip()
        text = match.group(2).strip()
        lang_map = {
            "french": "fr", "spanish": "es", "german": "de", "italian": "it", "japanese": "ja", "chinese": "zh-CN", "hindi": "hi", "arabic": "ar", "russian": "ru", "portuguese": "pt", "korean": "ko", "north korean": "ko", "english": "en"
        }
        lang_code = lang_map.get(target_language.lower(), "en")
        try:
            result = GoogleTranslator(source='auto', target=lang_code).translate(text)
            return {"response": result}
        except Exception as e:
            return {"error": str(e)}
    # Default: use local model
    payload = {
        "model": MODEL_NAME,
        "prompt": req.prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    return {"response": data.get("response", "")}

@app.post("/chat")
def chat_with_history(req: ChatRequest):
    # Build the prompt from the chat history
    prompt = ""
    for msg in req.history:
        if msg.role == "user":
            prompt += f"User: {msg.content}\n"
        else:
            prompt += f"Assistant: {msg.content}\n"
    prompt += "Assistant:"
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    bot_response = data.get("response", "")
    return {"response": bot_response}

@app.post("/chat-log/append")
def append_chat_log(req: ChatLogAppendRequest):
    try:
        with open(CHAT_LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {req.role.capitalize()}: {req.content}\n")
        return {"status": "ok"}
    except Exception as e:
        return {"status": "error", "detail": str(e)}

@app.get("/chat-log")
def get_chat_log():
    if not os.path.exists(CHAT_LOG_FILE):
        return Response(content="", media_type="text/plain")
    with open(CHAT_LOG_FILE, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content=content, media_type="text/plain")

@app.post("/chat-log/clear")
def clear_chat_log():
    open(CHAT_LOG_FILE, "w").close()
    return {"status": "cleared"}

@app.post("/data-test")
def data_test():
    import time
    prompts = [
        "What is the capital of France?",
        "Summarize the theory of relativity.",
        "List three benefits of regular exercise.",
        "Explain how photosynthesis works.",
        "Write a short poem about the ocean."
    ]
    results = []
    for prompt in prompts:
        start = time.time()
        payload = {
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload)
        elapsed = time.time() - start
        data = response.json()
        output = data.get("response", "")
        token_count = len(output.split())
        results.append({
            "prompt": prompt,
            "response_time_sec": round(elapsed, 2),
            "output_length": token_count
        })
    # Save to CSV
    df = pd.DataFrame(results)
    csv_path = "model_test_results.csv"
    df.to_csv(csv_path, index=False)

    # Plot response time
    plt.figure(figsize=(8,4))
    plt.bar(range(len(prompts)), [r["response_time_sec"] for r in results], tick_label=[f"Test {i+1}" for i in range(5)])
    plt.ylabel("Response Time (s)")
    plt.title("Ollama Model Response Time")
    plt.tight_layout()
    buf1 = io.BytesIO()
    plt.savefig(buf1, format="png")
    plt.close()
    buf1.seek(0)
    img1_b64 = base64.b64encode(buf1.read()).decode()

    # Plot output length
    plt.figure(figsize=(8,4))
    plt.bar(range(len(prompts)), [r["output_length"] for r in results], tick_label=[f"Test {i+1}" for i in range(5)])
    plt.ylabel("Output Length (words)")
    plt.title("Ollama Model Output Length")
    plt.tight_layout()
    buf2 = io.BytesIO()
    plt.savefig(buf2, format="png")
    plt.close()
    buf2.seek(0)
    img2_b64 = base64.b64encode(buf2.read()).decode()

    return {
        "results": results,
        "csv_path": csv_path,
        "response_time_plot": img1_b64,
        "output_length_plot": img2_b64
    }

@app.get("/data-test/download-csv")
def download_data_test_csv():
    return FileResponse("model_test_results.csv", media_type="text/csv", filename="model_test_results.csv")

@app.post("/summarize-pdf")
async def summarize_pdf(file: UploadFile = File(...)):
    # Read PDF file
    pdf_reader = PyPDF2.PdfReader(file.file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text() or ""
    # Optionally truncate text if too long for the model
    text = text[:4000]
    # Summarize with Ollama
    payload = {
        "model": MODEL_NAME,
        "prompt": f"Summarize the following PDF content concisely and do not add any follow-up questions, suggestions, or extra commentary. Only return the summary.\n{text}",
        "stream": False
    }
    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    data = response.json()
    return {"response": data.get("response", "")}

@app.post("/summarize-youtube")
def summarize_youtube(req: dict):
    youtube_url = req.get("youtube_url")
    if not youtube_url:
        return {"error": "No YouTube URL provided."}
    try:
        video_id = YouTube(youtube_url).video_id
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
        transcript_text = " ".join([entry['text'] for entry in transcript])
        # Optionally truncate transcript if too long for the model
        transcript_text = transcript_text[:4000]
        # Summarize with Ollama
        payload = {
            "model": MODEL_NAME,
            "prompt": f"Summarize the following YouTube transcript concisely and do not add any follow-up questions, suggestions, or extra commentary. Only return the summary.\n{transcript_text}",
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=payload)
        response.raise_for_status()
        data = response.json()
        summary = data.get("response", "")
        formatted = "\n".join([f"{entry['start']:.2f}s: {entry['text']}" for entry in transcript])
        return {"summary": summary, "transcript": formatted}
    except NoTranscriptFound:
        return {"error": "No transcript found for this video."}
    except TranscriptsDisabled:
        return {"error": "Transcripts are disabled for this video."}
    except Exception as e:
        return {"error": f"An error occurred: {e}"} 