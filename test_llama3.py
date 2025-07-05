# --- Imports ---
import os
import tempfile
import shutil
import logging
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
from datetime import datetime
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker

# --- Config ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3:latest"
EMBED_MODEL = "intfloat/e5-large-v2"

# --- Title ---
st.set_page_config(page_title="PDF QA App", layout="wide")
st.title("📄 Multi-PDF Q&A with Docling + HuggingFace")

# --- Upload Sidebar ---
st.sidebar.header("📁 Upload PDFs")
uploaded_files = st.sidebar.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

# --- Session States ---
if "file_paths" not in st.session_state:
    st.session_state.file_paths = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Save Uploaded Files ---
if uploaded_files:
    for f in uploaded_files:
        temp_path = os.path.join(tempfile.gettempdir(), f.name)
        with open(temp_path, "wb") as temp_file:
            temp_file.write(f.read())
        if temp_path not in st.session_state.file_paths:
            st.session_state.file_paths.append(temp_path)

# --- Display Files ---
if st.session_state.file_paths:
    st.sidebar.markdown("*Files to index:*")
    for path in st.session_state.file_paths:
        st.sidebar.checkbox(os.path.basename(path), value=True, key=path)

# --- Embedding Model ---
embed_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# --- Extract & Index ---
if st.sidebar.button("📊 Extract & Index"):
    all_docs = []
    for path in st.session_state.file_paths:
        try:
            loader = DoclingLoader(
                file_path=path,
                export_type=ExportType.DOC_CHUNKS,
                chunker=HybridChunker(tokenizer=EMBED_MODEL)
            )
            docs = loader.load()
            for d in docs:
                d.metadata['source'] = os.path.basename(path)
            all_docs.extend(docs)
        except Exception as e:
            st.error(f"❌ Failed to load {os.path.basename(path)}: {e}")

    if all_docs:
        vs = FAISS.from_documents(all_docs, embed_model)
        st.session_state.vectorstore = vs
        st.success("✅ Documents indexed successfully!")
    else:
        st.warning("⚠ No documents processed.")

# --- Chat UI ---
st.subheader("💬 Ask a question about your documents")
user_query = st.text_input("Enter your question")
if st.button("🔍 Submit Query") and user_query:
    if not st.session_state.vectorstore:
        st.warning("Please upload and index files first.")
    else:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})

        prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant. Answer the question based only on the provided context.

            Context:
            {context}

            Question:
            {input}

            Answer:
            """
        )

        llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_LLM_MODEL)
        chain = (
            {"context": retriever, "input": RunnablePassthrough()} 
            | prompt 
            | llm 
            | StrOutputParser()
        )

        response = chain.invoke(user_query)
        st.session_state.chat_history.append((user_query, response))

# --- Display Chat History ---
if st.session_state.chat_history:
    st.markdown("### 🧾 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1], 1):
        st.markdown(f"*Q{i}:* {q}")
        st.markdown(f"*A{i}:* {a}")