from pathlib import Path
from typing import List, Dict
from pypdf import PdfReader
from tqdm import tqdm


def load_documents(
    raw_dir: Path,
) -> List[Dict]:
    """
    Load documents from raw_dir.

    Returns:
        List of dicts with keys:
        - text: str
        - source: filename
    """
    documents = []

    for path in tqdm(raw_dir.glob("**/*.pdf")):
        try:
            reader = PdfReader(path)
            text=""
        
            for page in reader.pages:
                text += page.extract_text()
        
            documents.append({
                "text": text,
                "source": path.name,
            })
        except:
            continue

    return documents
