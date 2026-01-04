from typing import List, Dict

def chunk_document(
    document: Dict,
    chunk_size: int,
    overlap: int,
) -> List[Dict]:
    """
    Split a document into overlapping chunks.

    Returns:
        List of dicts with keys:
        - text
        - source
        - chunk_id
    """
    text = document["text"]
    words = text.split()

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]

        chunks.append({
            "text": " ".join(chunk_words),
            "source": document["source"],
            "chunk_id": chunk_id,
        })

        start += chunk_size - overlap
        chunk_id += 1

    return chunks
