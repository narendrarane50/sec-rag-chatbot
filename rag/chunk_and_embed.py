from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

import os
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

# ── Config ───────────────────────────────────────────────────
FILINGS_DIR  = Path("sec_filings")
CHROMA_DIR   = Path("chroma_db")
COLLECTION   = "sec_10k_filings"

CHUNK_SIZE    = 400   # tokens (approx — we split by words)
CHUNK_OVERLAP = 50    # words of overlap between chunks

EMBED_MODEL = "BAAI/bge-large-en-v1.5"  # best for financial text

# ── Chunking ─────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    words  = text.split()
    chunks = []
    start  = 0
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        if len(chunk.strip()) > 50:   # skip tiny fragments
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# ── Load all .txt filings ─────────────────────────────────────
def load_filings(filings_dir: Path) -> list[dict]:
    """Returns list of {ticker, filename, text} dicts."""
    docs = []
    for ticker_dir in sorted(filings_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        for txt_file in sorted(ticker_dir.glob("*.txt")):
            text = txt_file.read_text(encoding="utf-8")
            docs.append({
                "ticker":   ticker,
                "filename": txt_file.name,
                "text":     text,
            })
            print(f"  Loaded {txt_file.name}  ({len(text):,} chars)")
    return docs

# ── Build ChromaDB vector store ───────────────────────────────
def build_vector_store(docs: list[dict]):
    print(f"\nEmbedding model: {EMBED_MODEL}")
    print("(First run downloads ~1.3 GB — this may take a few minutes)\n")

    # HuggingFace embedding function built into ChromaDB
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))

    # Drop and recreate if exists (for clean reruns)
    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print(f"Deleted existing collection '{COLLECTION}'")

    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0

    for doc in tqdm(docs, desc="Embedding filings"):
        ticker   = doc["ticker"]
        filename = doc["filename"]
        chunks   = chunk_text(doc["text"], CHUNK_SIZE, CHUNK_OVERLAP)

        if not chunks:
            print(f"  [!] No chunks for {filename}")
            continue

        # ChromaDB add() in batches of 100 to avoid memory spikes
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            ids   = [f"{filename}__chunk_{i + j}" for j in range(len(batch))]
            metas = [{"ticker": ticker, "filename": filename, "chunk_index": i + j}
                     for j in range(len(batch))]

            collection.add(
                documents=ids,       # ChromaDB stores the ID separately
                ids=ids,
                metadatas=metas,
            )
            # Add actual text separately (documents param = the text to embed)
            collection.update(
                ids=ids,
                documents=batch,
            )

        total_chunks += len(chunks)
        print(f"  {filename}: {len(chunks)} chunks")

    return collection, total_chunks

# ── Smoke test: run a sample query ───────────────────────────
def smoke_test(collection):
    print("\n── Smoke test ──")
    query = "What was Apple's total revenue?"
    results = collection.query(
        query_texts=[query],
        n_results=3,
    )
    print(f"Query: '{query}'")
    for i, (doc, meta) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0]
    )):
        print(f"\n  Result {i+1} [{meta['ticker']} | {meta['filename']} | chunk {meta['chunk_index']}]")
        print(f"  {doc[:200]}...")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── Loading filings ──")
    docs = load_filings(FILINGS_DIR)
    print(f"\nLoaded {len(docs)} filings\n")

    collection, total = build_vector_store(docs)

    print(f"\n── Done ──")
    print(f"Total chunks embedded: {total:,}")
    print(f"Vector store saved to: {CHROMA_DIR}/")

    smoke_test(collection)