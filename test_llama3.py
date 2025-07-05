# --- Imports ---
import os
import json
import uuid
import shutil
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_docling.loader import ExportType

# --- Config ---
OLLAMA_LLM_MODEL = "llama3:latest"
OLLAMA_EMBED_MODEL = "nomic-embed-text:latest"
DB_DIR = "faiss_db"

# --- Streamlit UI ---
st.set_page_config(page_title="Simple PDF QA using Docling", layout="wide")
st.title("📄 Simple PDF QA using Docling")

# --- Session State ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- File Upload ---
with st.sidebar:
    st.header("📁 Upload PDFs")
    uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True)
    if uploaded:
        st.session_state.uploaded_files = uploaded
        for f in uploaded:
            st.markdown(f"✅ {f.name}")

    if st.button("📊 Extract & Index"):
        all_docs = []
        os.makedirs("temp", exist_ok=True)

        for f in st.session_state.uploaded_files:
            file_path = os.path.join("temp", f"{uuid.uuid4()}_{f.name}")
            with open(file_path, "wb") as out:
                out.write(f.read())

            loader = DoclingLoader(
                file_path=file_path,
                export_type=ExportType.DOC_CHUNKS,
                chunker=HybridChunker(
                    tokenizer="intfloat/e5-base",  # ✅ valid tokenizer
                    preserve_table_blocks=True,
                    max_tokens=1024
                )
            )

            try:
                docs = loader.load()
                for d in docs:
                    d.metadata["source"] = os.path.basename(file_path)
                all_docs.extend(docs)
            except Exception as e:
                st.error(f"❌ Failed to process {f.name}: {e}")

        if all_docs:
            embedder = OllamaEmbeddings(model=OLLAMA_EMBED_MODEL)
            if os.path.exists(DB_DIR):
                vs = FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
                vs.add_documents(all_docs)
            else:
                vs = FAISS.from_documents(all_docs, embedder)
            vs.save_local(DB_DIR)
            st.success("✅ Documents indexed successfully.")
        else:
            st.warning("⚠ No documents were indexed.")

    if st.button("🗑 Clear FAISS Index"):
        if os.path.exists(DB_DIR):
            shutil.rmtree(DB_DIR)
            st.success("🧹 Index cleared.")
        else:
            st.info("No existing index found.")

    if st.button("🧼 Clear Chat"):
        st.session_state.chat_history = []

# --- Load Vectorstore ---
def load_index():
    if not os.path.exists(DB_DIR):
        return None
    try:
        return FAISS.load_local(DB_DIR, OllamaEmbeddings(model=OLLAMA_EMBED_MODEL), allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load index: {e}")
        return None

vs = load_index()

# --- Chat Input ---
st.subheader("💬 Ask a question")
query = st.text_input("Type your question here")

if st.button("🔍 Ask") and query:
    if not vs:
        st.warning("Please upload and index documents first.")
    else:
        retriever = vs.as_retriever(search_kwargs={"k": 5})
        llm = ChatOllama(model=OLLAMA_LLM_MODEL)

        prompt = ChatPromptTemplate.from_template(
            "You are an assistant answering questions based on extracted document content.\n\nContext:\n{context}\n\nQuestion:\n{question}\n\nAnswer:"
        )

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.spinner("Thinking..."):
            answer = chain.invoke(query)
            st.session_state.chat_history.append((query, answer))
            st.success("✅ Answer generated.")

# --- Display Chat History ---
if st.session_state.chat_history:
    st.markdown("### 🧾 Chat History")
    for q, a in st.session_state.chat_history[::-1]:
        st.markdown(f"🧑‍💻 Q:** {q}")
        st.markdown(f"🤖 A:** {a}")