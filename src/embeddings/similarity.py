import numpy as np

def cosine_similarity(query_vec: np.ndarray, embeddings: np.ndarray):
    """
    Compute cosine similarity between query vector and all embeddings
    """
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query_vec)
    sims = (embeddings @ query_vec.T) / norms
    return sims

def retrieve_top_k(query_vec: np.ndarray, embeddings: np.ndarray, k=5):
    sims = cosine_similarity(query_vec, embeddings)
    top_k_idx = sims.argsort()[-k:][::-1]
    return top_k_idx
