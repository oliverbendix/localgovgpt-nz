import os
from pinecone import Pinecone

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
from pinecone import ServerlessSpec

index_name = "localgovgpt"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,  # for OpenAI embeddings
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",        # or "gcp"
            region="us-east-1"  # check Pinecone UI for your region
        )
    )
