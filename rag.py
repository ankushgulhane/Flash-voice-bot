"""Scope-guarded RAG: embed query, retrieve top-3 from ChromaDB, answer only from context."""
import os

import chromadb
from dotenv import load_dotenv
from groq import Groq
from sentence_transformers import SentenceTransformer

load_dotenv()

CHROMA_DIR = "chroma_db"
COLLECTION = "knowledge_base"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # multilingual: supports Hindi + English
LLM_MODEL = "llama-3.3-70b-versatile"
TOP_K = 3
DISTANCE_THRESHOLD = 0.85  # cosine distance; larger = less similar

# Per-language refusal strings and human-readable names for the reply-language instruction.
REFUSALS = {
    "en": "I'm sorry, I can only answer questions based on the knowledge base, and I couldn't find that information there.",
    "hi": "क्षमा करें, मैं केवल नॉलेज बेस में मौजूद जानकारी के आधार पर ही उत्तर दे सकता हूँ, और वह जानकारी मुझे वहाँ नहीं मिली।",
}
LANG_NAMES = {"en": "English", "hi": "Hindi"}


def _system_prompt(lang):
    refusal = REFUSALS.get(lang, REFUSALS["en"])
    lang_name = LANG_NAMES.get(lang, "English")
    return (
        "You are a helpful assistant named Flash that answers strictly from the provided context. "
        f"Always reply in {lang_name}. Use only the information in the context. "
        "Cite sources inline using [1], [2], [3]. "
        f"If the answer is not in the context, reply with EXACTLY this text and nothing else: {refusal}"
    )

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")

_embedder = SentenceTransformer(EMBED_MODEL)
_client = chromadb.PersistentClient(path=CHROMA_DIR)
_groq = Groq()


def _build_collection():
    """Build the collection from the committed docs in data/.

    Runs on fresh deploys (e.g. Hugging Face Spaces) where chroma_db/ is gitignored
    and `python ingest.py` was never run. Reuses the already-loaded embedder/client.
    """
    import ingest

    docs = ingest.read_documents(ingest.DATA_DIR)
    if not docs:
        raise RuntimeError(
            f"No documents found in '{ingest.DATA_DIR}/' to build the knowledge base."
        )

    chunks, metadatas, ids = [], [], []
    for source, text in docs:
        for i, chunk in enumerate(ingest.chunk_text(text)):
            chunks.append(chunk)
            metadatas.append({"source": source, "chunk": i})
            ids.append(f"{source}-{i}")
    if not chunks:
        raise RuntimeError("Documents were found but contained no extractable text.")

    embeddings = _embedder.encode(chunks, normalize_embeddings=True).tolist()
    collection = _client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)
    return collection


def _get_collection():
    try:
        return _client.get_collection(COLLECTION)
    except Exception:
        return _build_collection()


def ask(query, lang="en"):
    """Return (answer_text, source_chunks). Refuses if query is out of scope. lang: 'en' or 'hi'."""
    refusal = REFUSALS.get(lang, REFUSALS["en"])
    query = (query or "").strip()
    if not query:
        return refusal, []

    collection = _get_collection()
    q_emb = _embedder.encode([query], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=q_emb, n_results=TOP_K)

    docs = result.get("documents", [[]])[0]
    distances = result.get("distances", [[]])[0]
    metas = result.get("metadatas", [[]])[0]

    if not docs or distances[0] > DISTANCE_THRESHOLD:
        return refusal, []

    context = "\n\n".join(f"[{i + 1}] {doc}" for i, doc in enumerate(docs))
    sources = [
        {"id": i + 1, "source": m.get("source", "?"), "distance": round(d, 3), "text": doc}
        for i, (doc, m, d) in enumerate(zip(docs, metas, distances))
    ]

    completion = _groq.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        max_tokens=300,
        messages=[
            {"role": "system", "content": _system_prompt(lang)},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    )
    answer = completion.choices[0].message.content.strip()
    return answer, sources


if __name__ == "__main__":
    a, s = ask("How much does the Pro plan cost?")
    print(a)
    print(f"\n{len(s)} source(s) used.")
