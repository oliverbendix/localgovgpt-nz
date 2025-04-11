# =============================
# split_documents.py
# =============================

import os
import pickle
from langchain.docstore.document import Document
from langchain.text_splitter import TokenTextSplitter

RAW_DATA_DIR = "data/fetched"
OUTPUT_PATH = "data/split_docs.pkl"


def load_fetched_documents():
    documents = []
    for root, _, files in os.walk(RAW_DATA_DIR):
        for file in files:
            if file.endswith(".txt"):
                path = os.path.join(root, file)
                with open(path, "r") as f:
                    lines = f.readlines()
                    source = lines[0].split(":", 1)[1].strip() if lines[0].startswith("source:") else "unknown"
                    content = "".join(lines[2:])  # skip source and scraped_at
                    documents.append(Document(page_content=content, metadata={"source": source}))
    return documents


def split_documents():
    docs = load_fetched_documents()
    splitter = TokenTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"[💾] Saved {len(chunks)} split documents to {OUTPUT_PATH}")


if __name__ == "__main__":
    split_documents()

