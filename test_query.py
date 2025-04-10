import os
from pinecone import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings

# Init Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("localgovgpt")

# Your query
query = "How do I report illegal dumping?"

# Embed the query
embeddings_model = OpenAIEmbeddings()
query_vector = embeddings_model.embed_query(query)

# Query the index
results = index.query(vector=query_vector, top_k=5, include_metadata=True)

# Print results
print("\nTop Results:")
for match in results["matches"]:
    score = match["score"]
    metadata = match["metadata"]
    source = metadata.get("source", "unknown")
    print(f"\n📎 Source: {source}")
    print(f"🔢 Score: {score:.3f}")
