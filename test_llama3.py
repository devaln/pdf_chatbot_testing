# --- Imports ---
import os
import uuid
import shutil
import json
import streamlit as st
from pathlib import Path
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker

# --- Config ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3:latest"
HF_EMBED_MODEL = "intfloat/e5-base"
DB_DIR = "./faiss_db"
TOP_K = 10

# --- Embedding ---
embedder = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)

# --- App UI ---
st.set_page_config(page_title="PDF QA using Docling", layout="wide")
st.title("📄 PDF Q&A (Docling Based)")
st.markdown("Ask questions about PDF tables, rows, and text.")

# --- Session ---
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "vs" not in st.session_state:
    st.session_state.vs = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- FAISS Load Helper ---
def load_existing_index():
    index_path = Path(DB_DIR) / "index.faiss"
    if not index_path.exists():
        return None
    try:
        return FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None

# --- Docling Extract & Index ---
def process_and_index(files):
    docs = []
    os.makedirs("temp", exist_ok=True)
    for file in files:
        temp_path = os.path.join("temp", file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())

        loader = DoclingLoader(
            file_path=temp_path,
            export_type=ExportType.DOC_CHUNKS,
            chunker=HybridChunker(tokenizer="intfloat/e5-base", mode="table_first")
        )
        try:
            docs.extend(loader.load())
        except Exception as e:
            st.error(f"Docling failed on {file.name}: {e}")

    if not docs:
        return None

    texts = [d.page_content for d in docs]
    metadatas = [d.metadata for d in docs]
    valid_docs = [Document(page_content=texts[i], metadata=metadatas[i]) for i in range(len(texts))]

    if Path(DB_DIR).exists():
        vs = FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
        vs.add_documents(valid_docs)
    else:
        vs = FAISS.from_documents(valid_docs, embedder)
    vs.save_local(DB_DIR)
    return vs

# --- LLM Chain ---
def get_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    prompt = ChatPromptTemplate.from_template(
        """
        You are a smart PDF analysis assistant.

        - If the user asks about a column (e.g., 'number of persons benefitted'), return all values under that column from every row.
        - If the user asks about a specific row (e.g., mentioning a CSR project or a teacher/child group), match that row based on content and return its values.
        - Always extract complete structured data if available.

        Use only the context provided below:
        {context}

        Question: {question}

        Answer:
        """
    )
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploader = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    if uploader:
        st.session_state.uploaded_files = uploader
        for f in uploader:
            st.markdown(f"✅ {f.name}")

    if st.button("📄 Extract & Index"):
        if st.session_state.uploaded_files:
            with st.spinner("Processing PDFs..."):
                vs = process_and_index(st.session_state.uploaded_files)
                if vs:
                    st.session_state.vs = vs
                    st.session_state.msgs = []
                    st.success("✅ Indexed successfully!")
                    st.session_state.uploaded_files = []
        else:
            st.warning("Please upload PDFs first.")

    if st.button("🧹 Clear Chat"):
        st.session_state.msgs = []

    if st.button("🗑 Clear FAISS Index"):
        shutil.rmtree(DB_DIR, ignore_errors=True)
        st.session_state.vs = None
        st.success("🗑 FAISS index deleted.")

# --- Load Existing Index ---
if st.session_state.vs is None:
    st.session_state.vs = load_existing_index()

# --- Chat UI ---
st.markdown("### 💬 Ask your question")
query = st.chat_input("Type your question here...")

if query:
    if not st.session_state.vs:
        st.error("Please upload and index PDFs first.")
    else:
        st.session_state.msgs.append({"role": "user", "content": query})
        with st.spinner("Thinking..."):
            try:
                chain = get_chain(st.session_state.vs)
                result = chain.invoke(query)
                if not result or len(result.strip()) < 2:
                    result = "⚠ Sorry, I couldn't find relevant data. Try refining the question."
            except Exception as e:
                result = f"⚠ LLM fallback failed: {e}"
            st.session_state.msgs.append({"role": "assistant", "content": result})

# --- Chat Display ---
for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])