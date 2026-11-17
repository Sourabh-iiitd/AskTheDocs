import glob
import os
import pickle

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

DOCS_FOLDER = "docs"
INDEX_FILE = "index.pkl"
CHUNK_SIZE = 800
OVERLAP = 150


def load_documents(folder):
    pages = []

    # 1) PDFs that contain real text
    for path in glob.glob(os.path.join(folder, "*.pdf")):
        reader = PdfReader(path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({"source": os.path.basename(path), "page": i + 1, "text": text})

    # 2) Plain text files
    for path in glob.glob(os.path.join(folder, "*.txt")):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        if text.strip():
            pages.append({"source": os.path.basename(path), "page": 1, "text": text})

    return pages


def chunk_text(text, size, overlap):
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks


def main():
    pages = load_documents(DOCS_FOLDER)
    if not pages:
        print("No documents found in docs/ folder.")
        return

    chunks = []
    for p in pages:
        for piece in chunk_text(p["text"], CHUNK_SIZE, OVERLAP):
            chunks.append({"source": p["source"], "page": p["page"], "text": piece})
    print(len(pages), "pages ->", len(chunks), "chunks")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode([c["text"] for c in chunks], normalize_embeddings=True)

    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
    print("Saved index to", INDEX_FILE)


if __name__ == "__main__":
    main()