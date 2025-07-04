# --- Imports ---
import os
import json
import uuid
import tempfile
import shutil
import logging
import streamlit as st
import pandas as pd
import fitz
import pdfplumber
import pytesseract
import requests
from fuzzywuzzy import fuzz
from pdf2image import convert_from_path
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
import pandasql as psql

# --- Config ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3:latest"
DB_DIR = "./faiss_db"
TOP_K = 5

# --- Embedding ---
bge_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

class BGEEmbedder:
    def embed_documents(self, texts):
        return bge_model.encode(texts, normalize_embeddings=True).tolist()
    def embed_query(self, query):
        return bge_model.encode(query, normalize_embeddings=True).tolist()

embedder = BGEEmbedder()

# --- App UI ---
st.set_page_config(page_title="PDF QA with Fuzzy Tables + Pandas Agent", layout="wide")
st.title("📊 PDF QA: Table Matching + Smart Queries")

# --- Cache & Logging ---
logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s [%(levelname)s] %(message)s")

# --- Table Storage ---
if "table_store" not in st.session_state:
    st.session_state.table_store = []

def extract_text_from_pdf(path):
    try:
        doc = fitz.open(path)
        return "\n".join([page.get_text() for page in doc])
    except:
        return ""

def extract_text_from_scanned_pdf(path):
    try:
        images = convert_from_path(path, dpi=300)
        return "\n".join([pytesseract.image_to_string(img) for img in images])
    except:
        return ""

def ask_llm_for_json_tables(text):
    prompt = f"""
Extract all tables from the following text and return them as JSON objects in this format:

[
  {{
    "table_title": "title above table",
    "columns": ["col1", "col2"],
    "rows": [["r1", "r2"], ...]
  }}
]

Text:
{text}
"""
    try:
        res = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        return json.loads(res.json().get("response", "[]"))
    except Exception as e:
        logging.error(f"JSON parse failed: {e}")
        return []

def extract_all_tables(path, scanned=False):
    text = extract_text_from_scanned_pdf(path) if scanned else extract_text_from_pdf(path)

    table_objs = ask_llm_for_json_tables(text)
    combined_chunks = []

    for table in table_objs:
        title = table.get("table_title", "Unnamed Table")
        cols = table.get("columns", [])
        rows = table.get("rows", [])
        try:
            df = pd.DataFrame(rows, columns=cols)
            st.session_state.table_store.append({"title": title, "df": df})
            st.subheader(f"📌 {title}")
            st.dataframe(df)
            combined_chunks.append(f"Title: {title}\n{df.to_csv(index=False)}")
        except Exception as e:
            logging.warning(f"Bad table skipped: {e}")
    return "\n\n".join(combined_chunks), text
# --- FAISS Indexing ---
def load_and_index(files, scanned=False):
    from langchain.vectorstores import FAISS
    all_docs = []
    with tempfile.TemporaryDirectory() as td:
        for file in files:
            file_path = os.path.join(td, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            try:
                text = extract_text_from_pdf(file_path) if not scanned else extract_text_from_scanned_pdf(file_path)
                tables, _ = extract_all_tables(file_path, scanned)
                doc_text = f"{tables}\n\n{text}"
                all_docs.append(Document(page_content=doc_text, metadata={"source": file.name}))
            except Exception as e:
                st.error(f"{file.name} failed: {e}")

    if not all_docs:
        st.warning("No documents processed.")
        return None

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]
    vectors = embedder.embed_documents(texts)

    if os.path.exists(DB_DIR):
        faiss_index = FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
        faiss_index.add_texts(texts, metadatas)
    else:
        faiss_index = FAISS.from_texts(texts, embedder, metadatas=metadatas)

    faiss_index.save_local(DB_DIR)
    st.success("✅ Indexed successfully!")
    return faiss_index

def load_existing_index():
    if not os.path.exists(DB_DIR): return None
    try:
        return FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Index load error: {e}")
        return None

def get_chat_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    prompt = ChatPromptTemplate.from_template(
        "You are a helpful assistant for analyzing PDF content.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

def fuzzy_match_table(query):
    best = {"score": 0, "title": None, "df": None}
    for t in st.session_state.table_store:
        score = fuzz.partial_ratio(query.lower(), t["title"].lower())
        if score > best["score"]:
            best = {"score": score, "title": t["title"], "df": t["df"]}
    return best if best["score"] >= 70 else None

def run_pandas_agent(df, query):
    try:
        query_lower = query.lower()
        if "total" in query_lower or "sum" in query_lower:
            return df.sum(numeric_only=True).to_frame("Total")
        elif "average" in query_lower or "mean" in query_lower:
            return df.mean(numeric_only=True).to_frame("Average")
        elif "where" in query_lower:
            return psql.sqldf(f"SELECT * FROM df WHERE {query_lower.split('where')[1]}", locals())
        else:
            return df.head()
    except Exception as e:
        return f"❌ Pandas agent failed: {e}"

def clear_db():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        st.success("🗑 Index cleared.")

# --- Sidebar ---
with st.sidebar:
    st.image("img/ACL_Digital.png", width=180)
    st.image("img/Cipla_Foundation.png", width=180)
    st.markdown("---")
    st.header("📂 Upload PDFs")

    if "uploader_key" not in st.session_state:
        st.session_state.uploader_key = str(uuid.uuid4())

    uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True, key=st.session_state.uploader_key)
    scanned_mode = st.checkbox("📸 PDF is scanned?")
    if st.button("📊 Extract & Index"):
        if uploaded:
            st.session_state.msgs = []
            st.session_state.table_store = []
            with st.spinner("Processing..."):
                st.session_state.vs = load_and_index(uploaded, scanned_mode)
            st.session_state.uploader_key = str(uuid.uuid4())

    st.markdown("---")
    if st.button("🗑 Clear DB"):
        clear_db()
        st.session_state.vs = None
    if st.button("🧹 Clear Chat"):
        st.session_state.msgs = []

# --- Main Chat ---
if "vs" not in st.session_state:
    st.session_state.vs = load_existing_index()
if "msgs" not in st.session_state:
    st.session_state.msgs = []

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
            st.markdown(f"🔍 Matched Table: *{match['title']}*")
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