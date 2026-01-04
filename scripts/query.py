import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from src.embeddings.similarity import retrieve_top_k
from src.generation.prompt import build_prompt
from src.generation.generate import generate_answer

# Embeddings + chunks paths
CHUNKS_FILE = Path("data/processed/chunks.json")
EMBEDDINGS_FILE = Path("data/embeddings/chunks_embeddings.npy")

# Retrieval parameters
TOP_K = 5
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# def main():
#     # Load chunks
#     chunks = json.loads(CHUNKS_FILE.read_text())
#     print(f"Loaded {len(chunks)} chunks.")

#     # Load embeddings
#     embeddings = np.load(EMBEDDINGS_FILE)
#     print(f"Loaded embeddings: {embeddings.shape}")

#     # Initialize embedding model
#     model = SentenceTransformer(EMBEDDING_MODEL, device="mps")

#     # Take user query
#     query = input("Enter your query: ").strip()
#     query_vec = model.encode([query])[0]

#     # Retrieve top-k
#     top_idx = retrieve_top_k(query_vec, embeddings, k=TOP_K)
#     print(f"\nTop {TOP_K} chunks:\n" + "-"*40)
#     for rank, idx in enumerate(top_idx, 1):
#         chunk = chunks[idx]
#         print(f"Rank {rank} | Source: {chunk['source']} | Chunk ID: {chunk['chunk_id']}")
#         print(chunk['text'][:500] + "...\n")  # first 500 chars

def main():
    # Load chunks
    chunks = json.loads(CHUNKS_FILE.read_text())
    print(f"Loaded {len(chunks)} chunks.")

    # Load embeddings
    embeddings = np.load(EMBEDDINGS_FILE)
    print(f"Loaded embeddings: {embeddings.shape}")

    # Initialize embedding model
    model = SentenceTransformer(EMBEDDING_MODEL, device="mps")

    # Take user query
    query = input("Enter your query: ").strip()
    query_vec = model.encode([query])[0]

    top_idx = retrieve_top_k(query_vec, embeddings, k=TOP_K)
    retrieved_chunks = [chunks[i] for i in top_idx]

    # Build prompt
    prompt = build_prompt(query, retrieved_chunks)
    print("===== PROMPT SENT TO LLM =====")
    print(prompt)
    print("===== END PROMPT =====")
    # Generate answer
    answer = generate_answer(prompt)
    print("\n" + "="*50 + "\n")
    print(f"Answer:\n{answer}\n")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
