import os
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utilities.scraper import fetch_multiple

# Where to save the vector store
VECTOR_STORE_PATH = "data/vector_store"

# URLs to fetch
urls = [
    "https://www.aucklandcouncil.govt.nz/report-problem",
    "https://services.wellington.govt.nz/report/",
    "https://vote.nz/enrolling/enrol-or-update/enrol-to-vote/"
]

# Fetch and clean content
pages = fetch_multiple(urls)

# Create LangChain Documents with metadata
documents = []
for page in pages:
    documents.append(Document(
        page_content=page["text"],
        metadata={"source": page["url"]}
    ))

# Chunk documents
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(documents)

# Embed and store in FAISS
embeddings = OpenAIEmbeddings()
vectordb = FAISS.from_documents(split_docs, embeddings)
vectordb.save_local(VECTOR_STORE_PATH)

print(f"[✅] Vector store saved to: {VECTOR_STORE_PATH}")
