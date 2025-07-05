import os
import uuid
import shutil
import streamlit as st
import pandas as pd
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker
from sentence_transformers import SentenceTransformer

# ------------------------ Config ------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3:latest"
OLLAMA_EMBED_MODEL = "nomic-embed-text"
DB_DIR = "./faiss_docling"
TOP_K = 5

# --------------------- Embedder -------------------------
bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")
class BGEEmbedder:
    def embed_documents(self, texts):
        return bge_model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, query):
        return bge_model.encode(query, normalize_embeddings=True).tolist()
embedder = BGEEmbedder()

# ------------------- App UI Setup -----------------------
st.set_page_config(page_title="Simple PDF QA using Docling", layout="wide")
st.title("📄 Simple PDF QA using Docling")
st.markdown("Ask questions about uploaded scanned or digital PDFs below:")

if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "vs" not in st.session_state:
    st.session_state.vs = None

# ----------------- FAISS Safe Loader ---------------------
def load_existing_index():
    index_file = Path(DB_DIR) / "index.faiss"
    if not index_file.exists():
        return None
    try:
        return FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None

# ------------------- Docling Indexing --------------------
def index_with_docling(files):
    docs = []
    os.makedirs("temp", exist_ok=True)
    for file in files:
        temp_path = os.path.join("temp", file.name)
        with open(temp_path, "wb") as f:
            f.write(file.read())

        loader = DoclingLoader(
            file_path=temp_path,
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(tokenizer="intfloat/e5-base")
        )
        try:
            docs.extend(loader.load())
        except Exception as e:
            st.error(f"Failed to process {file.name}: {e}")
    if not docs:
        return None

    # Embed and save
    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    vectors = embedder.embed_documents(texts)

    valid_docs = [
        Document(page_content=texts[i], metadata=metadatas[i])
        for i in range(len(texts)) if vectors[i] and sum(vectors[i]) != 0
    ]

    if Path(DB_DIR).exists():
        vs = FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
        vs.add_documents(valid_docs)
    else:
        vs = FAISS.from_documents(valid_docs, embedder)

    vs.save_local(DB_DIR)
    return vs

# ------------------- LLM Chain Setup ---------------------
def get_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant for analyzing PDF content.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# ------------------- Sidebar UI --------------------------
with st.sidebar:
    st.subheader("📂 Upload PDFs")
    uploaded = st.file_uploader("Upload one or more PDFs", type="pdf", accept_multiple_files=True)
    if st.button("📄 Extract & Index"):
        if uploaded:
            with st.spinner("Processing..."):
                vs = index_with_docling(uploaded)
                if vs:
                    st.session_state.vs = vs
                    st.success("✅ Documents indexed.")
                    st.session_state.msgs.clear()
        else:
            st.warning("Please upload at least one PDF.")

    st.markdown("---")
    if st.button("🧹 Clear Chat"):
        st.session_state.msgs.clear()
    if st.button("🗑 Clear Index"):
        shutil.rmtree(DB_DIR, ignore_errors=True)
        st.session_state.vs = None
        st.success("FAISS index deleted.")

# ------------------- Chat Interface ----------------------
st.markdown("### 💬 Ask a question")
query = st.text_input("Type your question here")

if st.button("🔍 Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    elif not st.session_state.vs:
        st.warning("Please upload and index documents first.")
    else:
        st.session_state.msgs.append({"role": "user", "content": query})
        with st.spinner("Thinking..."):
            chain = get_chain(st.session_state.vs)
            result = chain.invoke(query)
            st.session_state.msgs.append({"role": "assistant", "content": result})

# ------------------- Chat History ------------------------
if st.session_state.msgs:
    st.markdown("### 📜 Chat History")
    for msg in st.session_state.msgs:
        role = msg["role"]
        icon = "🧑‍💻" if role == "user" else "🤖"
        with st.chat_message(role):
            st.markdown(f"{icon} *{'Q:' if role == 'user' else 'A:'}* {msg['content']}")