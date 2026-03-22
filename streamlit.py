import streamlit as st
import os
from src.rag.rag_pipeline import SimpleRAG

st.set_page_config(page_title="Study Chatbot", page_icon="📚")

st.title("📚 Multi-Subject Study Chatbot")

# Load subjects
subjects = os.listdir("data")
subject = st.selectbox("Select Subject", subjects)

# Initialize RAG per subject
if "rag" not in st.session_state or st.session_state.get("subject") != subject:
    with st.spinner("Loading subject..."):
        path = f"data/{subject}"
        st.session_state.rag = SimpleRAG(subject, path)
        st.session_state.subject = subject

# Question input
question = st.text_input("Ask your question")

if st.button("Ask") and question:
    answer, docs = st.session_state.rag.ask(question)

    st.markdown("### 💡 Answer")
    st.success(answer)

    with st.expander("📄 Retrieved Chunks"):
        for i, d in enumerate(docs, 1):
            st.text(f"{i}. {d.page_content[:300]}...")