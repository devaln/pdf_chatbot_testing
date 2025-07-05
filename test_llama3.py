# --- Imports ---
import os
import shutil
import uuid
import streamlit as st
import json
import pandas as pd
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker

# --- Config ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3:latest"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
DB_DIR = "faiss_db"
TOP_K = 5

st.set_page_config(page_title="Simple PDF QA", layout="wide")
st.title("\U0001F4C4 Simple PDF QA using Docling")

# --- Embedder ---
embedder = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL, base_url=OLLAMA_BASE_URL)

# --- Session ---
if "vs" not in st.session_state:
    st.session_state.vs = None
if "msgs" not in st.session_state:
    st.session_state.msgs = []

# --- FAISS Index Loader ---
def load_existing_index():
    try:
        return FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
    except:
        return None

# --- Indexing ---
def index_with_docling(files):
    docs = []
    os.makedirs("temp", exist_ok=True)
    for file in files:
        temp_path = os.path.join("temp", file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        loader = DoclingLoader(
            file_path=temp_path,
            export_type="DOC_CHUNKS",
            chunker=HybridChunker(tokenizer=OLLAMA_EMBED_MODEL),
        )
        d = loader.load()
        docs.extend(d)

    if not docs:
        st.warning("No valid docs to index.")
        return

    json_path = os.path.join("temp", "docling_chunks.json")
    with open(json_path, "w") as f:
        json.dump([doc.dict() for doc in docs], f, indent=2)

    if os.path.exists(DB_DIR):
        vs = FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
        vs.add_documents(docs)
    else:
        vs = FAISS.from_documents(docs, embedder)

    vs.save_local(DB_DIR)
    st.success("\u2705 Indexed with Docling!")
    return vs

# --- LLM Chain ---
def get_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    prompt = ChatPromptTemplate.from_template(
        """You are a helpful assistant. Use the context below to answer the question.
        
        Context:
        {context}

        Question: {question}
        Answer:"""
    )
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# --- Sidebar UI ---
with st.sidebar:
    st.header("\U0001F4C2 Upload PDFs")
    uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)

    if st.button("\U0001F4CA Extract & Index") and uploaded:
        st.session_state.vs = index_with_docling(uploaded)
        st.session_state.msgs = []

    if st.button("\U0001F5D1 Clear Index"):
        shutil.rmtree(DB_DIR, ignore_errors=True)
        st.session_state.vs = None
        st.success("Cleared!")

    if st.button("\U0001F9F9 Clear Chat"):
        st.session_state.msgs = []

# --- Load existing DB ---
if st.session_state.vs is None:
    st.session_state.vs = load_existing_index()

# --- Chat UI ---
st.subheader("\U0001F5E8 Ask Questions")
for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask your question about uploaded PDFs..."):
    st.session_state.msgs.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vs:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = get_chain(st.session_state.vs)
                result = chain.invoke(query)
                st.markdown(result)
                st.session_state.msgs.append({"role": "assistant", "content": result})
    else:
        st.error("Please upload and index first.")