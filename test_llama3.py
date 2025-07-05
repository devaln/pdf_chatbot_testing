# --- Imports ---
import os
import tempfile
import shutil
import streamlit as st
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableMap
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker

# --- Config ---
OLLAMA_LLM_MODEL = "llama3:latest"
OLLAMA_EMBED_MODEL = "nomic-embed-text"

DB_DIR = "faiss_index"
os.makedirs(DB_DIR, exist_ok=True)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# --- Streamlit Setup ---
st.set_page_config(page_title="Simple PDF QA using Docling", layout="wide")
st.title("📄 Simple PDF QA using Docling")
st.markdown("Ask questions about uploaded scanned or digital PDFs below:")

# --- Upload Section ---
uploaded_files = st.file_uploader("📁 Upload PDF files", type=["pdf"], accept_multiple_files=True)
if uploaded_files:
    temp_files = []
    for uploaded in uploaded_files:
        temp_path = os.path.join(tempfile.gettempdir(), uploaded.name)
        with open(temp_path, "wb") as f:
            f.write(uploaded.read())
        temp_files.append(temp_path)
else:
    temp_files = []

# --- Session Setup ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# --- Embedding Model ---
embedder = HuggingFaceBgeEmbeddings(
    model_name=OLLAMA_EMBED_MODEL,
    encode_kwargs={"normalize_embeddings": True}
)

# --- Indexing Logic ---
def index_with_docling(file_paths):
    all_docs = []
    for path in file_paths:
        loader = DoclingLoader(
            file_path=path,
            export_type=ExportType.CHUNKS,
            chunker=HybridChunker(tokenizer="BAAI/bge-base-en")
        )
        docs = loader.load()
        for d in docs:
            d.metadata["source"] = os.path.basename(path)
        all_docs.extend(docs)

    # Load or create FAISS DB
    if os.path.exists(os.path.join(DB_DIR, "index.faiss")):
        vs = FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
        vs.add_documents(all_docs)
    else:
        vs = FAISS.from_documents(all_docs, embedder)
    vs.save_local(DB_DIR)
    return vs

# --- Index Button ---
if temp_files and st.button("📊 Extract & Index"):
    vs = index_with_docling(temp_files)
    st.session_state.vectorstore = vs
    st.success("✅ Documents indexed successfully!")

# --- Chat UI ---
st.markdown("### 💬 Ask a question")
query = st.text_input("Type your question here")

if st.button("🔍 Ask") and query:
    if not st.session_state.vectorstore:
        st.warning("Please upload and index documents first.")
    else:
        retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 5})
        llm = ChatOllama(model=OLLAMA_LLM_MODEL, temperature=0)

        prompt = ChatPromptTemplate.from_template(
            """
You are an intelligent assistant helping extract accurate information from documents.
Answer clearly based only on the context below. If the answer is a table, return it cleanly.

Context:
{context}

Question:
{input}

Answer:
"""
        )

        chain = (
            RunnableMap({"context": retriever, "input": RunnablePassthrough()})
            | prompt
            | llm
            | StrOutputParser()
        )

        try:
            result = chain.invoke(query)
            st.session_state.chat_history.append((query, result))
            st.success("✅ Answer generated.")
        except Exception as e:
            st.error(f"❌ Error generating answer: {e}")

# --- Chat History ---
if st.session_state.chat_history:
    st.markdown("### 📜 Chat History")
    for i, (q, a) in enumerate(st.session_state.chat_history[::-1], 1):
        st.markdown(f"*Q{i}:* {q}")
        st.markdown(f"*A{i}:* {a}")