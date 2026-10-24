import json
import os

import requests
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

router = APIRouter()

GEMINI_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
STUDIO_KEY = os.environ.get("STUDIO_KEY", "")
API_BASE = "https://generativelanguage.googleapis.com/v1beta/models/"


class TranscribeIn(BaseModel):
    audio_b64: str
    mime: str = "audio/wav"
    language: str = ""


def _check(key: str):
    if not STUDIO_KEY or key != STUDIO_KEY:
        raise HTTPException(status_code=401, detail="Wrong or missing studio password")
    if not GEMINI_KEY:
        raise HTTPException(status_code=500, detail="The server has no GEMINI_API_KEY set")


def _seconds(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    total = 0.0
    try:
        for part in str(value).strip().split(":"):
            total = total * 60 + float(part)
    except ValueError:
        return 0.0
    return total


@router.get("/api/ai/health")
def health():
    return {
        "ok": True,
        "gemini_key_set": bool(GEMINI_KEY),
        "studio_password_set": bool(STUDIO_KEY),
        "model": GEMINI_MODEL,
    }


@router.post("/api/ai/transcribe")
def transcribe(body: TranscribeIn, x_studio_key: str = Header(default="")):
    _check(x_studio_key)
    prompt = (
        "Transcribe the speech in this audio into short caption segments. "
        "Each segment is one phrase of at most 12 words. "
        "Return ONLY a JSON array. Each item has: "
        '"start" (MM:SS.s), "end" (MM:SS.s) and "text". '
        "Do not describe music and do not add speaker names."
    )
    if body.language.strip():
        prompt += " The speech is in " + body.language.strip() + "."
    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": body.mime, "data": body.audio_b64}},
                ]
            }
        ],
        "generationConfig": {"responseMimeType": "application/json", "temperature": 0.1},
    }
    r = requests.post(
        API_BASE + GEMINI_MODEL + ":generateContent",
        headers={"x-goog-api-key": GEMINI_KEY, "Content-Type": "application/json"},
        json=payload,
        timeout=180,
    )
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail="Gemini answered %d: %s" % (r.status_code, r.text[:300]))
    try:
        text = r.json()["candidates"][0]["content"]["parts"][0]["text"]
        data = json.loads(text)
    except Exception:
        raise HTTPException(status_code=502, detail="Could not read the transcript")
    if isinstance(data, dict):
        data = data.get("segments", [])
    segments = []
    for item in data:
        t = str(item.get("text", "")).strip()
        if t:
            segments.append(
                {"start": _seconds(item.get("start", 0)), "end": _seconds(item.get("end", 0)), "text": t}
            )
    return {"segments": segments}
