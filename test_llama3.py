# --- Imports ---
import os
import uuid
import shutil
import logging
import streamlit as st
import pandas as pd
from fuzzywuzzy import fuzz
import pandasql as psql

from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough

from langchain_docling import DoclingLoader
from langchain_docling.loader import ExportType
from docling.chunking import HybridChunker

# --- Config ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3:latest"
DB_DIR = "./faiss_db"
TOP_K = 5

# --- Embedding Wrapper ---
class HuggingFaceEmbedder:
    def _init_(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer("e5-large-v2")

    def embed_documents(self, texts):
        return self.model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, query):
        return self.model.encode(query, normalize_embeddings=True).tolist()

embedder = HuggingFaceEmbedder()

# --- App UI ---
st.set_page_config(page_title="PDF QA with Docling", layout="wide")
st.title("\U0001F4C4 PDF QA: Table Matching + Smart Queries")

# --- Logging ---
logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s [%(levelname)s] %(message)s")

# --- Session Init ---
if "table_store" not in st.session_state:
    st.session_state.table_store = []
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- FAISS Indexing ---
def load_and_index(files):
    all_docs = []
    with st.spinner("Processing with Docling..."):
        for file in files:
            temp_path = os.path.join("temp", file.name)
            os.makedirs("temp", exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())

            loader = DoclingLoader(
                file_path=temp_path,
                export_type=ExportType.DOC_CHUNKS,
                chunker=HybridChunker(tokenizer="intfloat/e5-large-v2")
            )

            try:
                docs = loader.load()
                all_docs.extend(docs)
            except Exception as e:
                st.error(f"Failed to load {file.name}: {e}")

    if not all_docs:
        st.warning("No documents processed.")
        return None

    texts = [d.page_content for d in all_docs]
    metadatas = [d.metadata for d in all_docs]
    vectors = embedder.embed_documents(texts)

    clean_docs = [Document(page_content=texts[i], metadata=metadatas[i])
                  for i in range(len(texts)) if vectors[i] and sum(vectors[i]) != 0]

    if os.path.exists(DB_DIR):
        vs = FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
        vs.add_documents(clean_docs)
    else:
        vs = FAISS.from_documents(clean_docs, embedder)

    vs.save_local(DB_DIR)
    st.success("\u2705 Indexed successfully!")
    return vs

# --- Safe FAISS Loader ---
def load_existing_index():
    faiss_file = os.path.join(DB_DIR, "index.faiss")
    if not os.path.exists(DB_DIR) or not os.path.exists(faiss_file):
        return None
    try:
        return FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Index load error: {e}")
        return None

# --- LLM Chain ---
def get_chat_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant for analyzing PDF content.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# --- Table Matching ---
def fuzzy_match_table(query):
    best = {"score": 0, "title": None, "df": None}
    for t in st.session_state.table_store:
        score = fuzz.partial_ratio(query.lower(), t["title"].lower())
        if score > best["score"]:
            best = {"score": score, "title": t["title"], "df": t["df"]}
    return best if best["score"] >= 70 else None

# --- Pandas Agent ---
def run_pandas_agent(df, query):
    try:
        query_lower = query.lower()
        if "total" in query_lower or "sum" in query_lower:
            return df.sum(numeric_only=True).to_frame("Total")
        elif "average" in query_lower or "mean" in query_lower:
            return df.mean(numeric_only=True).to_frame("Average")
        elif "count" in query_lower:
            return df.count(numeric_only=True).to_frame("Count")
        elif "max" in query_lower:
            return df.max(numeric_only=True).to_frame("Max")
        elif "min" in query_lower:
            return df.min(numeric_only=True).to_frame("Min")
        elif "where" in query_lower:
            return psql.sqldf(f"SELECT * FROM df WHERE {query_lower.split('where')[1]}", locals())
        else:
            return df.head()
    except Exception as e:
        return f"\u274c Pandas agent failed: {e}"

# --- DB Clear ---
def clear_db():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        st.success("\U0001f5d1 Index cleared.")

# --- Sidebar ---
with st.sidebar:
    st.image("img/ACL_Digital.png", width=180)
    st.image("img/Cipla_Foundation.png", width=180)
    st.markdown("---")
    st.header("\U0001F4C2 Upload PDFs")

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = str(uuid.uuid4())

    uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True, key=st.session_state.uploader_key)
    if uploaded:
        st.session_state.uploaded_files = uploaded
        st.markdown("#### Files to index:")
        for file in uploaded:
            st.markdown(f"\u2705 {file.name}")

    if st.button("\U0001F4CA Extract & Index"):
        if st.session_state.uploaded_files:
            st.session_state.msgs = []
            st.session_state.table_store = []
            st.session_state.vs = load_and_index(st.session_state.uploaded_files)
            st.session_state.uploader_key = str(uuid.uuid4())
            st.session_state.uploaded_files = []

    st.markdown("---")
    if st.button("\U0001f5d1 Clear DB"):
        clear_db()
        st.session_state.vs = None
    if st.button("\U0001f9f9 Clear Chat"):
        st.session_state.msgs = []

# --- Main Chat ---
if "vs" not in st.session_state:
    st.session_state.vs = load_existing_index()

for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about a table (e.g., revenue from finance table)..."):
    st.session_state.msgs.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    match = fuzzy_match_table(query)
    if match:
        with st.chat_message("assistant"):
            st.markdown(f"\U0001F50D Matched Table: *{match['title']}*")
            result = run_pandas_agent(match["df"], query)
            if isinstance(result, str):
                st.markdown(result)
            else:
                st.dataframe(result)
    elif st.session_state.vs:
        chain = get_chat_chain(st.session_state.vs)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = "".join(chain.stream(query))
                st.markdown(response)
                st.session_state.msgs.append({"role": "assistant", "content": response})
    else:
        st.error("Please upload and index documents first.")