"""Turn an uploaded PDF into overlapping text chunks ready for embedding."""
import io

from pypdf import PdfReader


def extract_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """Simple sliding-window chunker over whitespace-normalized text.

    chunk_size / overlap are in characters, not tokens -- good enough for a
    demo. Overlap keeps a sentence that straddles a chunk boundary from being
    split with no context on either side.
    """
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks = []
    start = 0
    n = len(normalized)
    step = max(chunk_size - overlap, 1)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start += step
    return chunks


def pdf_to_documents(pdf_bytes: bytes, source_name: str,
                      chunk_size: int = 800, overlap: int = 150) -> list[dict]:
    text = extract_text(pdf_bytes)
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=overlap)
    return [
        {"id": f"{source_name}-chunk-{i}", "text": chunk}
        for i, chunk in enumerate(chunks)
    ]
