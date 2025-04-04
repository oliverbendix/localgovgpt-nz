import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from utils.load_documents import load_and_embed_docs

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load civic docs and build vector DB (temporary in memory for now)
retriever = load_and_embed_docs()

llm = ChatOpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

def get_civic_answer(question: str):
    result = qa_chain(question)
    answer = result["result"]
    sources = [doc.metadata.get("source", "Unknown") for doc in result["source_documents"]]
    return {"answer": answer, "sources": sources}
