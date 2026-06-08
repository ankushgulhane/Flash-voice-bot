# Voice RAG Demo (MVP)

A cost-effective, learning-focused voice assistant that runs locally for ~₹0.

**Flow:** you speak → speech-to-text → RAG retrieval over a local knowledge base with a
scope guardrail → LLM answers **only** from context → text-to-speech reply, wrapped in a Gradio web UI.

## Tech stack (all free tier)

| Component   | Choice                                              |
|-------------|-----------------------------------------------------|
| STT         | Groq Whisper (`whisper-large-v3-turbo`), cloud      |
| LLM         | Groq Llama 3.3 70B (`llama-3.3-70b-versatile`)      |
| Embeddings  | `all-MiniLM-L6-v2` (sentence-transformers, local)   |
| Vector DB   | ChromaDB (local, file-based persistent client)      |
| TTS         | `edge-tts` (`en-IN-NeerjaNeural`), local            |
| UI          | Gradio (mic input + autoplay audio output)          |

Only STT and the LLM call the cloud (Groq's free tier). Everything else runs on your CPU.

## Prerequisites

- Python 3.10+
- A free Groq API key
- A working microphone and speakers

## Get a free Groq API key

1. Go to <https://console.groq.com>.
2. Sign in (Google/GitHub or email).
3. Open **API Keys** → **Create API Key**, name it, and copy the value.
4. Groq's free tier is generous and requires no credit card.

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure your API key
cp .env.example .env
# then edit .env and set GROQ_API_KEY=your_real_key
```

## Run

```bash
# Build the knowledge base from everything in data/
python ingest.py

# Launch the voice UI (opens http://127.0.0.1:7860)
python app.py
```

Click the microphone, ask a question (e.g. *"How much does the Pro plan cost?"*),
and the assistant answers by voice using only the documents in `data/`.

## Add your own knowledge

Drop `.txt` or `.pdf` files into `data/` and re-run `python ingest.py`.
Ingestion is idempotent — it resets the `kb` collection each run.

## How to test the scope guardrail

The assistant must refuse anything not in the knowledge base.

- **In scope** (answers): *"What is the return policy?"*, *"How much is the Team plan?"*
- **Out of scope** (refuses): *"What's the weather in Paris?"*, *"Who won the World Cup?"*

For out-of-scope questions you should hear the fixed refusal:

> *I'm sorry, I can only answer questions based on the knowledge base, and I couldn't find that information there.*

Tune sensitivity via `DISTANCE_THRESHOLD` in `rag.py` (default `0.8` cosine distance).
Lower it to refuse more aggressively; raise it to be more permissive.

## Project layout

```
.
├── .env.example     # GROQ_API_KEY=your_key_here
├── requirements.txt
├── ingest.py        # data/ docs → chunk → embed → ChromaDB "kb"
├── rag.py           # scope check + top-3 retrieval + LLM with citations
├── voice.py         # transcribe() via Groq Whisper, speak() via edge-tts
├── app.py           # Gradio UI (entry point)
├── data/            # knowledge base (sample.txt included)
└── README.md
```

## Notes on latency

End-to-end target is ~1s. The local embedding and TTS steps are fast; the two Groq
calls dominate. The first run downloads the embedding model (~90 MB) and is slower.
