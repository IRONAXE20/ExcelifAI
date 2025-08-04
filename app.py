import streamlit as st
import os
from dotenv import load_dotenv
from rag_utils.summarizer import summarize_text
from rag_utils.parse_pdf import extract_text_from_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from rag_utils.rag_chain import build_qa_chain

st.set_page_config(page_title="ExcelifAI", layout="wide")
load_dotenv()

st.title("ExcelifAI - Your AI Study Assistant")

uploaded_files = st.file_uploader("Upload your PDF study materials", type="pdf", accept_multiple_files=True)

if uploaded_files:
    all_texts = []
    all_docs = []
    for file in uploaded_files:
        save_path = os.path.join("uploads", file.name)
        with open(save_path, "wb") as f:
            f.write(file.read())
        st.success(f"Uploaded and saved: {file.name}")
        text = extract_text_from_pdf(save_path)
        all_texts.append(text)

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text)

        for chunk in chunks:
            all_docs.append(Document(page_content=chunk, metadata={"source": file.name}))

    st.subheader("AI Summaries")
    for i, text in enumerate(all_texts):
        st.markdown(f"### Summary for: `{uploaded_files[i].name}`")
        summary = summarize_text(text)
        st.write(summary)

    qa_chain = build_qa_chain(all_docs)

    st.subheader("🔍 Ask Questions About the Material")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    query = st.text_input("Type your question here")

    if query:
        with st.spinner("Searching..."):
            result = qa_chain.invoke({
                "question": query,
                "chat_history": st.session_state.chat_history
            })
            answer = result["answer"]

        st.success("Answer:")
        st.write(answer)
        st.session_state.chat_history.append(("User", query))
        st.session_state.chat_history.append(("AI", answer))

    if st.session_state.chat_history:
        st.subheader("Conversation History")
        for speaker, msg in st.session_state.chat_history:
            st.markdown(f"**{speaker}:** {msg}")
