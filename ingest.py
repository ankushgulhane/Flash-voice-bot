"""Load docs from data/, chunk, embed, and store in a ChromaDB collection 'kb'."""
import glob
import os

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DATA_DIR = "data"
CHROMA_DIR = "chroma_db"
COLLECTION = "knowledge_base"
EMBED_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # must match rag.py
CHUNK_WORDS = 350      # ~500 tokens
OVERLAP_WORDS = 35     # ~50 tokens


def read_documents(data_dir):
    docs = []
    for path in sorted(glob.glob(os.path.join(data_dir, "*"))):
        ext = os.path.splitext(path)[1].lower()
        if ext == ".txt":
            with open(path, "r", encoding="utf-8") as f:
                docs.append((os.path.basename(path), f.read()))
        elif ext == ".pdf":
            reader = PdfReader(path)
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            docs.append((os.path.basename(path), text))
    return docs


def chunk_text(text, size=CHUNK_WORDS, overlap=OVERLAP_WORDS):
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        end = start + size
        chunk = " ".join(words[start:end]).strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


def main():
    docs = read_documents(DATA_DIR)
    if not docs:
        raise SystemExit(f"No .txt or .pdf files found in '{DATA_DIR}/'. Add documents and retry.")

    chunks, metadatas, ids = [], [], []
    for source, text in docs:
        for i, chunk in enumerate(chunk_text(text)):
            chunks.append(chunk)
            metadatas.append({"source": source, "chunk": i})
            ids.append(f"{source}-{i}")

    if not chunks:
        raise SystemExit("Documents were found but contained no extractable text.")

    print(f"Embedding {len(chunks)} chunks from {len(docs)} document(s)...")
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(chunks, show_progress_bar=True, normalize_embeddings=True).tolist()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        client.delete_collection(COLLECTION)  # idempotent: reset on re-run
    except Exception:
        pass
    collection = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)

    print(f"Ingested {len(chunks)} chunks into collection '{COLLECTION}' at ./{CHROMA_DIR}")


if __name__ == "__main__":
    main()
