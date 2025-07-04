# --- Imports ---
import os
import json
import tempfile
import shutil
import logging
import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import pdfplumber
import camelot
import pytesseract
import requests
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough

# --- Config ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3:latest"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DB_DIR = "./faiss_db"

st.set_page_config(page_title="PDF QA with JSON Tables", layout="wide")
st.title("📊 PDF JSON Table Extractor + QA Chat")

logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s [%(levelname)s] %(message)s")

# --- Helpers ---
def extract_text_from_pdf(path):
    try:
        doc = fitz.open(path)
        return "\n".join([page.get_text() for page in doc])
    except Exception as e:
        logging.warning(f"Text extraction failed: {e}")
        return ""

def ask_llm_for_json_tables(text):
    prompt = f"""
You are a table extraction expert.

Extract all tables from the text below. Return the result as a JSON list of objects like:

[
  {{
    "table_title": "optional title above the table",
    "columns": ["col1", "col2", ...],
    "rows": [["val1", "val2", ...], ...]
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
        output = res.json().get("response", "")
        return json.loads(output)
    except Exception as e:
        logging.error(f"Failed to parse LLM JSON: {e}")
        return []

def extract_from_scanned_pdf(path):
    try:
        images = convert_from_path(path, dpi=300)
        text = ""
        for img in images:
            ocr_text = pytesseract.image_to_string(img)
            text += ocr_text + "\n"
        return text
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        return ""

def extract_all_tables(path, scanned_mode=False):
    if scanned_mode:
        text = extract_from_scanned_pdf(path)
    else:
        text = extract_text_from_pdf(path)
        try:
            with pdfplumber.open(path) as pdf:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        if table:
                            df = pd.DataFrame(table[1:], columns=table[0])
                            text += "\n" + df.to_csv(index=False)
        except:
            pass

    table_objs = ask_llm_for_json_tables(text)
    table_chunks = []
    for table in table_objs:
        title = table.get("table_title", "")
        cols = table.get("columns", [])
        rows = table.get("rows", [])
        try:
            df = pd.DataFrame(rows, columns=cols)
            st.subheader(f"📌 {title or 'Unnamed Table'}")
            st.dataframe(df)
            chunk = f"Title: {title}\n\n{df.to_csv(index=False)}"
            table_chunks.append(chunk)
        except Exception as e:
            logging.warning(f"Bad table skipped: {e}")

    return "\n\n".join(table_chunks), text

@st.cache_resource(show_spinner=False)
def load_existing_index():
    if not os.path.exists(DB_DIR): return None
    try:
        embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        return FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    except:
        return None

def load_and_index(files, scanned_mode=False):
    all_docs = []
    with tempfile.TemporaryDirectory() as td:
        for file in files:
            file_path = os.path.join(td, file.name)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
            try:
                loader = PyPDFLoader(file_path)
                all_docs.extend(loader.load())
                tables, text = extract_all_tables(file_path, scanned_mode)
                combined = f"{tables}\n\n{text}"
                all_docs.append(Document(page_content=combined, metadata={"source": file.name}))
            except Exception as e:
                st.error(f"{file.name} failed: {e}")

    if not all_docs:
        st.warning("No data extracted.")
        return None

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
    embeddings = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)

    if os.path.exists(DB_DIR):
        vs = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
        vs.add_documents(chunks)
    else:
        vs = FAISS.from_documents(chunks, embeddings)

    vs.save_local(DB_DIR)
    st.success("✅ Indexed successfully!")
    return vs

def get_chat_chain(vs):
    prompt = ChatPromptTemplate.from_template("You are a JSON table QA expert.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    return {"context": vs.as_retriever(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

def clear_db():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
        st.success("🗑 FAISS index cleared.")

# --- Sidebar ---
with st.sidebar:
    st.image("img/ACL_Digital.png", width=180)
    st.image("img/Cipla_Foundation.png", width=180)
    st.markdown("---")
    st.header("📂 Upload PDFs")
    uploaded = st.file_uploader("Upload PDF(s)", type="pdf", accept_multiple_files=True, key="upload_key")
    scanned_mode = st.checkbox("📸 PDF is scanned?")
    if st.button("📊 Extract & Index"):
        if uploaded:
            st.session_state.msgs = []
            with st.spinner("Processing and indexing..."):
                st.session_state.vs = load_and_index(uploaded, scanned_mode)
            st.session_state["upload_key"] = None  # clear file list from sidebar

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

if query := st.chat_input("Ask about tables or data..."):
    st.session_state.msgs.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vs:
        chain = get_chat_chain(st.session_state.vs)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = "".join(chain.stream(query))
                st.markdown(answer)
                st.session_state.msgs.append({"role": "assistant", "content": answer})
    else:
        st.error("Please upload and index documents first.")