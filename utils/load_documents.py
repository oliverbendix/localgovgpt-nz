import os
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document

def load_and_embed_docs():
    docs_path = "data/fetched/"
    documents = []

    for fname in os.listdir(docs_path):
        if not fname.endswith(".txt"):
            continue

        with open(os.path.join(docs_path, fname), "r") as f:
            lines = f.readlines()

            # Extract source from first line
            source_line = lines[0].strip()
            if source_line.lower().startswith("source:"):
                source_url = source_line.split(":", 1)[1].strip()
                content = "".join(lines[1:])  # skip source line
            else:
                source_url = "unknown"
                content = "".join(lines)

            documents.append(Document(
                page_content=content,
                metadata={"source": source_url}
            ))

    # Split and embed
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(split_docs, embeddings)

    return vectordb.as_retriever()

