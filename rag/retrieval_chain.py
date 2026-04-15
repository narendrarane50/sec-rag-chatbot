from bs4 import XMLParsedAsHTMLWarning
import warnings
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions
# transformers not needed — using Ollama for inference
import torch

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # hide CUDA
import torch
torch.backends.mps.enabled = False        # explicitly disable MPS
import requests as http_requests

# ── Config ───────────────────────────────────────────────────
CHROMA_DIR  = Path("chroma_db")
COLLECTION  = "sec_10k_filings"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

# Use base Mistral for now — swap this path once fine-tuning is done
OLLAMA_MODEL = "mistral"

TOP_K = 5  # number of chunks to retrieve per query

# ── Load vector store ─────────────────────────────────────────
def load_collection():
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client     = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION, embedding_function=ef)
    print(f"Loaded collection '{COLLECTION}' with {collection.count()} chunks")
    return collection

# ── Retrieve relevant chunks ──────────────────────────────────
def retrieve(collection, query: str, ticker: str = None, top_k: int = TOP_K):
    """Retrieve top-k chunks, optionally filtered by ticker."""
    where = {"ticker": ticker} if ticker else None
    results = collection.query(
        query_texts=[query],
        n_results=top_k,
        where=where,
    )
    chunks = []
    for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
        chunks.append({
            "text":     doc,
            "ticker":   meta["ticker"],
            "filename": meta["filename"],
            "chunk":    meta["chunk_index"],
        })
    return chunks

# ── Build prompt ──────────────────────────────────────────────
def build_prompt(query: str, chunks: list[dict]) -> str:
    context_parts = []
    for i, c in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}: {c['ticker']} | {c['filename']}]\n{c['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""<s>[INST] You are a financial analyst assistant. Answer the question using ONLY the provided SEC filing excerpts below. If the answer is not in the excerpts, say "I could not find that information in the provided filings."

Always cite which company and filing your answer comes from.

--- SEC FILING EXCERPTS ---
{context}
--- END EXCERPTS ---

Question: {query} [/INST]"""
    return prompt

# ── Load LLM ─────────────────────────────────────────────────
# def load_llm(model_path: str = LLM_MODEL):
#     print(f"\nLoading LLM: {model_path}")
#     print("(This may take a few minutes on first run...)\n")

#     tokenizer = AutoTokenizer.from_pretrained(model_path)

#     # Load in 4-bit if GPU available, else float32 on CPU
#     if torch.cuda.is_available():
#         from transformers import BitsAndBytesConfig
#         bnb_config = BitsAndBytesConfig(
#             load_in_4bit=True,
#             bnb_4bit_compute_dtype=torch.float16,
#         )
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             quantization_config=bnb_config,
#             device_map="auto",
#         )
#     else:
#         # CPU fallback — slow but works for testing
#         model = AutoModelForCausalLM.from_pretrained(
#             model_path,
#             torch_dtype=torch.float32,
#             device_map="cpu",
#         )

#     pipe = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_new_tokens=512,
#         temperature=0.1,       # low temp for factual financial answers
#         do_sample=True,
#         repetition_penalty=1.1,
#     )
#     print("LLM loaded.")
#     return pipe

# def load_llm(model_path: str = LLM_MODEL):
#     print(f"\nLoading LLM: {model_path}")
#     print("(Running on CPU — each response takes 3–8 mins, normal for Mac)\n")

#     tokenizer = AutoTokenizer.from_pretrained(model_path)

#     model = AutoModelForCausalLM.from_pretrained(
#         model_path,
#         dtype=torch.float32,
#         device_map="cpu",       # force CPU, skip MPS
#         low_cpu_mem_usage=True,
#     )

#     # pipe = pipeline(
#     #     "text-generation",
#     #     model=model,
#     #     tokenizer=tokenizer,
#     #     max_new_tokens=256,     # shorter = faster on CPU
#     #     temperature=0.1,
#     #     do_sample=True,
#     #     repetition_penalty=1.1,
#     # )

#     pipe = pipeline(
#     "text-generation",
#     model=model,
#     tokenizer=tokenizer,
#     max_new_tokens=256,
#     temperature=0.1,
#     do_sample=True,
#     repetition_penalty=1.1,
#     device=-1,              # -1 forces CPU explicitly
#     )
#     print("LLM loaded.")
#     return pipe

def load_llm():
    print("Using Ollama (Mistral via Metal GPU)")
    try:
        http_requests.get("http://localhost:11434", timeout=3)
        print("Ollama is running.\n")
    except Exception:
        print("[!] Ollama not reachable — make sure 'ollama serve' is running\n")
    return None

# ── Full RAG query ────────────────────────────────────────────
# def rag_query(query: str, collection, pipe, ticker: str = None):
#     # 1. Retrieve
#     chunks = retrieve(collection, query, ticker=ticker)

#     # 2. Build prompt
#     prompt = build_prompt(query, chunks)

#     # 3. Generate
#     output   = pipe(prompt)[0]["generated_text"]
#     answer   = output.split("[/INST]")[-1].strip()

#     return {
#         "query":   query,
#         "answer":  answer,
#         "sources": chunks,
#     }

def rag_query(query: str, collection, pipe, ticker: str = None):
    chunks = retrieve(collection, query, ticker=ticker)
    prompt = build_prompt(query, chunks)

    resp = http_requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "mistral",
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "repeat_penalty": 1.1,
                "num_predict": 256,
            }
        },
        timeout=120,
    )
    answer = resp.json().get("response", "").strip()

    return {
        "query":   query,
        "answer":  answer,
        "sources": chunks,
    }

def print_result(result: dict):
    print(f"\n{'─'*60}")
    print(f"Q: {result['query']}")
    print(f"\nA: {result['answer']}")
    print(f"\nSources:")
    for i, s in enumerate(result["sources"]):
        print(f"  [{i+1}] {s['ticker']} | {s['filename']} | chunk {s['chunk']}")
    print(f"{'─'*60}")

# ── Main: interactive CLI ─────────────────────────────────────
if __name__ == "__main__":
    collection = load_collection()
    pipe       = load_llm()

    print("\n── SEC Filing RAG Chatbot ──")
    print("Type 'quit' to exit. Optionally filter by ticker e.g. 'AAPL: what was revenue?'\n")

    while True:
        user_input = input("You: ").strip()
        if not user_input or user_input.lower() == "quit":
            break

        # Parse optional ticker prefix: "AAPL: what was revenue?"
        ticker = None
        query  = user_input
        if ":" in user_input:
            parts  = user_input.split(":", 1)
            prefix = parts[0].strip().upper()
            if prefix in ["AAPL","MSFT","GOOGL","AMZN","META","NVDA","TSLA","JPM","BAC","GS"]:
                ticker = prefix
                query  = parts[1].strip()

        result = rag_query(query, collection, pipe, ticker=ticker)
        print_result(result)