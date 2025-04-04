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

        st.markdown("---")
        st.markdown("### 📄 Sources")
        for i, source in enumerate(response["sources"], 1):
            st.markdown(f"**{i}.** {source}")
