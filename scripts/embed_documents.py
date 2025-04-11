import os
import pickle
import uuid
from pinecone import Pinecone, ServerlessSpec
from langchain_community.embeddings import OpenAIEmbeddings

INPUT_PATH = "data/split_docs.pkl"
INDEX_NAME = "localgovgpt"
BATCH_SIZE = 100


def embed_and_upload():
    with open(INPUT_PATH, "rb") as f:
        docs = pickle.load(f)

    embeddings_model = OpenAIEmbeddings()
    texts = [doc.page_content for doc in docs]
    metadatas = [doc.metadata for doc in docs]
    print("[🧠] Generating embeddings...")
    embeddings = embeddings_model.embed_documents(texts)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    if INDEX_NAME not in pc.list_indexes().names():
        pc.create_index(
            name=INDEX_NAME,
            dimension=1536,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )

    index = pc.Index(INDEX_NAME)
    print("[📤] Uploading to Pinecone...")

    for i in range(0, len(embeddings), BATCH_SIZE):
        batch = [
            (str(uuid.uuid4()), embeddings[i], metadatas[i])
            for i in range(i, min(i + BATCH_SIZE, len(embeddings)))
        ]
        index.upsert(vectors=batch)

    print(f"[✅] Uploaded {len(embeddings)} vectors to Pinecone.")


if __name__ == "__main__":
    embed_and_upload()
