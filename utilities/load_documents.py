import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

VECTOR_STORE_PATH = "data/vector_store"

def load_and_embed_docs():
    embeddings = OpenAIEmbeddings()

    if os.path.exists(os.path.join(VECTOR_STORE_PATH, "index.faiss")):
        print("[📦] Loading vector store from disk...")
        vectordb = FAISS.load_local(
            VECTOR_STORE_PATH,
            embeddings,
            allow_dangerous_deserialization=True  
        )
    else:
        raise FileNotFoundError(f"Vector store not found at {VECTOR_STORE_PATH}. Run the build script first.")

    return vectordb.as_retriever()


