"""Speech I/O: transcribe via Groq Whisper, synthesize via edge-tts."""
import asyncio
import os
import tempfile

import edge_tts
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

STT_MODEL = "whisper-large-v3-turbo"
TTS_VOICES = {"en": "en-IN-NeerjaNeural", "hi": "hi-IN-SwaraNeural"}

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

_groq = Groq()


def transcribe(audio_path, language="en"):
    """Transcribe an audio file to text using Groq Whisper. language: 'en' or 'hi'."""
    if not audio_path:
        return ""
    with open(audio_path, "rb") as f:
        result = _groq.audio.transcriptions.create(
            file=(os.path.basename(audio_path), f.read()),
            model=STT_MODEL,
            language=language,
        )
    return result.text.strip()


async def _synthesize(text, out_path, voice):
    await edge_tts.Communicate(text, voice).save(out_path)


def speak(text, lang="en"):
    """Synthesize text to an mp3 file and return its path. lang: 'en' or 'hi'."""
    if not text or not text.strip():
        return None
    voice = TTS_VOICES.get(lang, TTS_VOICES["en"])
    out_path = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False).name
    asyncio.run(_synthesize(text, out_path, voice))
    return out_path


if __name__ == "__main__":
    path = speak("Hello! This is the voice RAG demo speaking.")
    print(f"Saved sample speech to {path}")
