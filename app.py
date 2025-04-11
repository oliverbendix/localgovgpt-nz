import streamlit as st
from civicgpt_chain import get_civic_answer

st.set_page_config(page_title="CivicGPT NZ", page_icon="🇳🇿", layout="centered")

st.title("🇳🇿 LocalGovGPT NZ")
st.subheader("Ask a local government question:")
user_question = st.text_input("Your question", placeholder="e.g. How do I report a broken footpath in Auckland?")

if user_question:
    with st.spinner("Finding an answer..."):
        response = get_civic_answer(user_question)

    st.markdown("### 🧠 Answer")
    st.write(response["answer"])

    if response["sources"]:
        st.markdown("---")
        st.markdown("### 🔗 Sources consulted")
        for i, source in enumerate(response["sources"], 1):
            st.markdown(f"**{i}.** [{source}]({source})")
    else:
        st.info("No sources found for this response.")

st.markdown("---")
with st.container():
    st.markdown("### 👨‍💻 About this app")
    st.info(
"""
        **LocalGovGPT NZ** is a personal project built by [Oliver Thompson](mailto:oliverthompsoncv@gmail.com) 
        to explore the use of LLMs and Retrieval-Augmented Generation (RAG) in the public sector.

        It scrapes and indexes 78 New Zealand local council websites to help answer questions about local government services and policies.

        I built this because I'm keen to see AI adopted responsibly and safely in Aotearoa, and I think using tools is the best way to really understand them.

        If you're looking for an information, data and AI leader in 2026, please check out my other app [Oliver's CV Chatbot](https://cv-chatbot-oliverbendixthompson.streamlit.app) and feel free to get in touch!

        **Tech stack:** This app is built using Python, Streamlit, LangChain, OpenAI, and Pinecone. It uses asynchronous webcrawling, PDF/text parsing, and vector embeddings to power a lightweight civic LLM interface.
        """
    )
