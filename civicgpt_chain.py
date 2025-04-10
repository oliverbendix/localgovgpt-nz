import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from utilities.load_documents import load_and_embed_docs
import os
from pinecone import Pinecone
from langchain_community.embeddings import OpenAIEmbeddings
from openai import OpenAI

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

def get_civic_answer(question, top_k=5):
    embeddings = OpenAIEmbeddings()
    query_vector = embeddings.embed_query(question)

    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index = pc.Index("localgovgpt")
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)

    top_chunks = []
    sources = set()

    for match in results["matches"]:
        metadata = match["metadata"]
        content = metadata.get("text", "")
        source = metadata.get("source", "unknown")
        top_chunks.append(content)
        sources.add(source)

    context = "\n---\n".join(top_chunks[:3])

    # ✅ New OpenAI v1.0 style
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You're a helpful assistant answering questions using local government information in New Zealand."},
            {"role": "user", "content": f"Question: {question}\n\nUse the following:\n{context}"}
        ]
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": list(sources)
    }
