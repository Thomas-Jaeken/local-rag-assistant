import json
import numpy as np
from pathlib import Path
from src.embeddings.embed import embed_chunks

CHUNKS_FILE = Path("data/processed/chunks.json")
EMBEDDINGS_FILE = Path("data/embeddings/chunks_embeddings.npy")

def main():
    chunks = json.loads(CHUNKS_FILE.read_text())
    embeddings = embed_chunks(chunks)
    EMBEDDINGS_FILE.parent.mkdir(exist_ok=True)
    np.save(EMBEDDINGS_FILE, embeddings)
    print(f"Saved embeddings to {EMBEDDINGS_FILE}")

if __name__ == "__main__":
    main()
