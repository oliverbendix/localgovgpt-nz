import os
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings

def load_and_embed_docs():
    docs_path = "data/sample_docs/"
    loaders = [
        TextLoader(os.path.join(docs_path, fname)) 
        for fname in os.listdir(docs_path) 
        if fname.endswith(".txt")
    ]

    documents = []
    for loader in loaders:
        documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(split_docs, embeddings)

    return vectordb.as_retriever()
