"""
MediLex India — Database builder (BNS/BNSS/BSA edition).

Reads all .txt files from data/raw/, chunks them by section
delimiters, embeds with MiniLM, and stores in ChromaDB.

Run:
    python scripts/build_database.py

What it does:
  1. Reads each .txt file in data/raw/ — one file per Indian law
     (bns.txt, bnss.txt, bsa.txt, pocso.txt, jj_act.txt, mtp_act.txt).
  2. Chunks each file by "---" section delimiters (how the raw statute
     texts are already formatted).  Falls back to sliding-window
     chunking if no delimiters are found.
  3. Extracts section numbers from chunk text via regex (e.g.
     "Section 124" → stored as metadata for citation).
  4. Embeds all chunks with SentenceTransformer (all-MiniLM-L6-v2).
  5. Stores everything in a persistent ChromaDB collection.

The resulting chroma_db/ directory is what rag/retriever.py reads
at runtime.  Re-run this script whenever the raw statute files change.
"""

import os
import re
import sys
import logging

import chromadb
from sentence_transformers import SentenceTransformer

# Allow running from project root: `python scripts/build_database.py`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = "data/raw"
CHUNK_SIZE = 800   # sliding-window fallback only
OVERLAP = 80


def load_raw_texts() -> list[dict]:
    """Read every .txt in data/raw/ and return [{law, text}, ...]."""
    docs = []
    for filename in sorted(os.listdir(RAW_DIR)):
        if not filename.endswith(".txt"):
            continue
        law_name = filename.replace(".txt", "").upper()
        path = os.path.join(RAW_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        docs.append({"law": law_name, "text": text})
        logger.info(f"Loaded: {law_name} — {len(text):,} chars")
    return docs


def section_chunk(text: str) -> list[str]:
    """Split on --- delimiters (preferred for structured legal texts)."""
    parts = [p.strip() for p in text.split("---") if p.strip()]
    merged, buf = [], ""
    for p in parts:
        buf = (buf + "\n\n" + p).strip() if buf else p
        if len(buf) >= 150:
            merged.append(buf)
            buf = ""
    if buf:
        merged.append(buf)
    return merged


def sliding_chunk(text: str, size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[str]:
    """Fallback sliding-window chunker."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def chunk_text(text: str) -> list[str]:
    if "---" in text:
        return section_chunk(text)
    return sliding_chunk(text)


def extract_section(chunk: str) -> str | None:
    """Try to extract a section number from a chunk's text."""
    match = re.search(r"(?:Section|Sec\.?)\s*([A-Za-z0-9\-\(\)]+)", chunk, re.IGNORECASE)
    if not match:
        match = re.search(r"\b([0-9]+[A-Za-z]?(\([0-9A-Za-z]+\))?)\b", chunk)
    return match.group(1) if match else None


def build_db(docs: list[dict]) -> None:
    logger.info("Loading embedding model (first run downloads ~80 MB)...")
    model = SentenceTransformer(config.EMBEDDING_MODEL)

    client = chromadb.PersistentClient(path=config.CHROMA_PATH)
    try:
        client.delete_collection(config.COLLECTION_NAME)
        logger.info("Cleared old collection.")
    except Exception:
        pass

    collection = client.create_collection(
        name=config.COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    BATCH = 50
    batch_docs, batch_ids, batch_meta = [], [], []
    total = 0

    for doc in docs:
        chunks = chunk_text(doc["text"])
        logger.info(f"  {doc['law']}: {len(chunks)} chunks")
        for i, chunk in enumerate(chunks):
            section = extract_section(chunk)
            batch_docs.append(chunk)
            batch_ids.append(f"{doc['law']}_{i}")
            batch_meta.append({
                "law": doc["law"],
                "section": section,
                "chunk_index": i,
            })
            total += 1

            if len(batch_docs) >= BATCH:
                embeddings = model.encode(batch_docs).tolist()
                collection.add(
                    documents=batch_docs,
                    embeddings=embeddings,
                    ids=batch_ids,
                    metadatas=batch_meta,
                )
                batch_docs, batch_ids, batch_meta = [], [], []

    if batch_docs:
        embeddings = model.encode(batch_docs).tolist()
        collection.add(
            documents=batch_docs,
            embeddings=embeddings,
            ids=batch_ids,
            metadatas=batch_meta,
        )

    logger.info(f"Database built — {total} chunks stored across {len(docs)} laws.")
    logger.info(f"Laws indexed: {[d['law'] for d in docs]}")


if __name__ == "__main__":
    docs = load_raw_texts()
    if not docs:
        logger.warning("No .txt files found in data/raw/. Exiting.")
    else:
        build_db(docs)
