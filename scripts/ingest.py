from pathlib import Path
import json

from src.ingestion.load_documents import load_documents
from src.ingestion.chunking import chunk_document

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/processed")

def main():
    docs = load_documents(RAW_DIR)

    all_chunks = []
    for doc in docs:
        chunks = chunk_document(
            doc,
            chunk_size=150,
            overlap=20,
        )
        all_chunks.extend(chunks)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "chunks.json", "w") as f:
        json.dump(all_chunks, f, indent=2)

if __name__ == "__main__":
    main()
