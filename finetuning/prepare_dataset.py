import json
import random
from pathlib import Path
from tqdm import tqdm
import requests as http_requests

# ── Config ────────────────────────────────────────────────────
FILINGS_DIR = Path("sec_filings")
OUTPUT_DIR  = Path("finetuning/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE  = OUTPUT_DIR / "sec_qa_dataset.jsonl"
TARGET_PAIRS = 500   # target Q&A pairs total
CHUNK_SIZE   = 600   # words per context chunk fed to Ollama

# Question templates per financial topic
# Ollama will generate the answer from the actual filing text
QUESTION_TEMPLATES = [
    "What was the total revenue for {ticker} in this period?",
    "What was the net income for {ticker}?",
    "What were the main risk factors disclosed by {ticker}?",
    "How did operating expenses change for {ticker}?",
    "What was {ticker}'s cash and cash equivalents position?",
    "What were {ticker}'s main business segments?",
    "How did {ticker} describe its competitive landscape?",
    "What was {ticker}'s earnings per share?",
    "What capital expenditures did {ticker} report?",
    "How did {ticker} describe its growth strategy?",
    "What were {ticker}'s total assets?",
    "What long-term debt did {ticker} carry?",
    "How did gross margin change for {ticker}?",
    "What dividends did {ticker} pay?",
    "What acquisitions did {ticker} make?",
]

# ── Load and chunk filings ─────────────────────────────────────
def load_chunks(filings_dir: Path, chunk_size: int) -> list[dict]:
    chunks = []
    for ticker_dir in sorted(filings_dir.iterdir()):
        if not ticker_dir.is_dir():
            continue
        ticker = ticker_dir.name
        for txt_file in sorted(ticker_dir.glob("*.txt")):
            text  = txt_file.read_text(encoding="utf-8")
            words = text.split()
            # Non-overlapping chunks for dataset generation
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i : i + chunk_size])
                if len(chunk.strip()) > 100:
                    chunks.append({
                        "ticker":   ticker,
                        "filename": txt_file.name,
                        "text":     chunk,
                    })
    print(f"Loaded {len(chunks)} chunks from {filings_dir}")
    return chunks

# ── Generate Q&A pair via Ollama ──────────────────────────────
def generate_qa(ticker: str, context: str, question: str) -> dict | None:
    prompt = f"""You are a financial analyst. Given the following excerpt from {ticker}'s SEC 10-K filing, answer the question concisely and accurately. If the answer is not in the excerpt, respond with "NOT_FOUND".

SEC Filing Excerpt:
{context}

Question: {question}

Answer:"""

    try:
        resp = http_requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "mistral",
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 200,
                }
            },
            timeout=60,
        )
        answer = resp.json().get("response", "").strip()

        # Skip if model couldn't find the answer in this chunk
        if "NOT_FOUND" in answer or len(answer) < 20:
            return None

        # Format as instruction-following pair for SFTTrainer
        return {
            "instruction": question,
            "input":       f"Context from {ticker} 10-K filing:\n{context}",
            "output":      answer,
            "ticker":      ticker,
            "source":      ticker,
        }
    except Exception as e:
        print(f"  [!] Ollama error: {e}")
        return None

# ── Main generation loop ──────────────────────────────────────
def generate_dataset(chunks: list[dict], target: int) -> list[dict]:
    dataset  = []
    # Shuffle chunks for variety
    random.shuffle(chunks)

    # Track per-ticker counts for balance
    ticker_counts: dict[str, int] = {}
    max_per_ticker = target // 10  # ~50 per company for 10 companies

    with tqdm(total=target, desc="Generating Q&A pairs") as pbar:
        for chunk in chunks:
            if len(dataset) >= target:
                break

            ticker = chunk["ticker"]
            if ticker_counts.get(ticker, 0) >= max_per_ticker:
                continue

            # Pick a random question template for this chunk
            question = random.choice(QUESTION_TEMPLATES).format(ticker=ticker)

            pair = generate_qa(ticker, chunk["text"], question)
            if pair:
                dataset.append(pair)
                ticker_counts[ticker] = ticker_counts.get(ticker, 0) + 1
                pbar.update(1)
                pbar.set_postfix({"ticker": ticker, "total": len(dataset)})

    return dataset

# ── Save as JSONL ─────────────────────────────────────────────
def save_dataset(dataset: list[dict], output_file: Path):
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            # Remove metadata fields before saving
            record = {
                "instruction": item["instruction"],
                "input":       item["input"],
                "output":      item["output"],
            }
            f.write(json.dumps(record) + "\n")
    print(f"\nSaved {len(dataset)} pairs to {output_file}")

# ── Stats ─────────────────────────────────────────────────────
def print_stats(dataset: list[dict]):
    from collections import Counter
    tickers = Counter(d["source"] for d in dataset)
    print("\n── Dataset stats ──")
    print(f"Total pairs:  {len(dataset)}")
    print(f"Avg output length: {sum(len(d['output']) for d in dataset) // len(dataset)} chars")
    print("\nPairs per company:")
    for ticker, count in sorted(tickers.items()):
        print(f"  {ticker}: {count}")

# ── Main ──────────────────────────────────────────────────────
if __name__ == "__main__":
    print("── SEC 10-K Fine-tuning Dataset Generator ──")
    print("Make sure 'ollama serve' is running\n")

    chunks  = load_chunks(FILINGS_DIR, CHUNK_SIZE)
    dataset = generate_dataset(chunks, TARGET_PAIRS)

    if dataset:
        save_dataset(dataset, OUTPUT_FILE)
        print_stats(dataset)
        print(f"\nNext step: upload {OUTPUT_FILE} to Kaggle for QLoRA fine-tuning")
    else:
        print("[!] No pairs generated — is Ollama running?")