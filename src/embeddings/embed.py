from typing import List, Dict
import numpy as np
from sentence_transformers import SentenceTransformer

def embed_chunks(chunks: List[Dict], model_name="all-MiniLM-L6-v2") -> np.ndarray:
    """
    Converts a list of text chunks into embeddings.

    Args:
        chunks: list of dicts with 'text' keys
        model_name: huggingface model

    Returns:
        np.ndarray of shape (num_chunks, embedding_dim)
    """
    model = SentenceTransformer(model_name, device="mps")
    texts = [chunk["text"] for chunk in chunks]
    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return embeddings
