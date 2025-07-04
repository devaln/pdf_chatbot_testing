# --- Imports ---
import os
import tempfile
import shutil
import logging
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
import camelot
import pytesseract
from pdf2image import convert_from_path
from difflib import SequenceMatcher

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
CHAT_DIR = "./chat_sessions"

logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s [%(levelname)s] %(message)s")
st.set_page_config(page_title="PDF QA with Tables", layout="wide")
st.title("📄 PDF Text & Table Extractor + Chat QA")

# --- Styles ---
st.markdown("""
    <style>
        section[data-testid="stSidebar"] {
            background-color: white !important;
            border-right: 2px solid #e0e0e0 !important;
        }
    </style>
""", unsafe_allow_html=True)

# --- Helpers ---
def clean_df(df):
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    return df.fillna("")

def df_to_readable_text(df, title=""):
    lines = [f"Table Title: {title.strip()}\n"] if title else []
    for _, row in df.iterrows():
        row_text = "; ".join([f"{col.strip()}: {str(row[col]).strip()}" for col in df.columns])
        lines.append(f"Row: {row_text}")
    return "\n".join(lines)

def extract_tables_pdfplumber(pdf_path):
    blocks = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text_lines = page.extract_text().split("\n") if page.extract_text() else []
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df = clean_df(df)
                        title = ""
                        for i, line in enumerate(text_lines):
                            if all(col.strip() in line for col in table[0] if col):
                                if i > 0:
                                    title = text_lines[i - 1].strip()
                                break
                        blocks.append(df_to_readable_text(df, title))
    except Exception as e:
        logging.warning(f"pdfplumber failed: {e}")
    return blocks

def extract_tables_camelot(pdf_path):
    blocks = []
    for flavor in ["lattice", "stream"]:
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor=flavor)
            for t in tables:
                df = t.df
                if df.shape[0] > 1 and df.shape[1] > 1:
                    df = clean_df(df)
                    blocks.append(df_to_readable_text(df))
        except Exception as e:
            logging.warning(f"camelot {flavor} failed: {e}")
    return blocks

def extract_scanned_pdf_with_ocr(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        full_dfs = []
        for img in images:
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
            ocr_data = ocr_data.dropna(subset=["text"])
            ocr_data = ocr_data[ocr_data['conf'].astype(int) > 40]
            grouped = ocr_data.groupby(['block_num', 'par_num', 'line_num'])
            rows = [group.sort_values("left")["text"].tolist() for _, group in grouped]
            if not rows:
                continue
            max_cols = max(len(row) for row in rows)
            padded = [row + [""] * (max_cols - len(row)) for row in rows]
            df_raw = pd.DataFrame(padded).replace("", pd.NA).dropna(how="all").fillna("")
            header_row_idx = df_raw.apply(lambda row: row.str.len().gt(1).sum(), axis=1).idxmax()
            headers = df_raw.iloc[header_row_idx].tolist()
            headers = [h if h.strip() else "column" for h in headers]
            headers = pd.io.parsers.ParserBase({'names': headers})._maybe_dedup_names(headers)
            df_data = df_raw.iloc[header_row_idx + 1:].reset_index(drop=True)
            df_data.columns = headers
            full_dfs.append(df_data)
        if not full_dfs:
            return "", ""
        combined = pd.concat(full_dfs).reset_index(drop=True)
        return df_to_readable_text(combined), combined.to_string(index=False)
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        return "", ""

def extract_all_tables(pdf_path, scanned_mode=False):
    try:
        if scanned_mode:
            return extract_scanned_pdf_with_ocr(pdf_path)
        dfs = extract_tables_pdfplumber(pdf_path)
        dfs += extract_tables_camelot(pdf_path)
        full_text = ""
        try:
            doc = fitz.open(pdf_path)
            full_text = "\n".join([page.get_text() for page in doc])
        except Exception as e:
            logging.error(f"Text extraction failed: {e}")
        return "\n\n".join(dfs), full_text
    except Exception as e:
        logging.error(f"Table extraction failed: {e}")
        return "", ""

def load_existing_index():
    if not os.path.exists(os.path.join(DB_DIR, "index.faiss")):
        return None
    try:
        embed = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        return FAISS.load_local(DB_DIR, embed, allow_dangerous_deserialization=True)
    except Exception as e:
        logging.error(f"Failed to load FAISS index: {e}")
        return None

def load_and_index(files, scanned_mode=False):
    embed = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    all_docs = []
    with tempfile.TemporaryDirectory() as td:
        for file in files:
            path = os.path.join(td, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            try:
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())
                tables_text, full_text = extract_all_tables(path, scanned_mode)
                if tables_text.strip():
                    doc_text = f"From file: {file.name}\n{tables_text}\n\n{full_text}"
                    all_docs.append(Document(page_content=doc_text, metadata={"source": file.name}))
            except Exception as e:
                logging.error(f"Failed to process {file.name}: {e}")
                st.error(f"Failed to process {file.name}: {e}")
    if not all_docs:
        st.warning("No documents were successfully extracted.")
        return st.session_state.get("vs", None)
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
    if not chunks:
        st.warning("No valid content to index.")
        return st.session_state.get("vs", None)
    if "vs" in st.session_state and st.session_state.vs:
        vs = st.session_state.vs
        vs.add_documents(chunks)
    else:
        vs = FAISS.from_documents(chunks, embed)
    vs.save_local(DB_DIR)
    return vs

def get_chat_chain(vs):
    prompt = ChatPromptTemplate.from_template(
        """
        You are a document and table analysis expert.

        Use only the information provided in the context below.
        If a table contains exact numbers or targets, include them precisely.
        Never mention document IDs or internal references like 'Document(id=...)'.

        Context:
        {context}

        Question: {question}

        Answer:
        """
    )
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    return {"context": vs.as_retriever(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# --- Sidebar UI ---
st.sidebar.image("img/ACL_Digital.png", width=180)
st.sidebar.image("img/Cipla_Foundation.png", width=180)
st.sidebar.markdown(""" <hr> """, unsafe_allow_html=True)

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

st.sidebar.header("📂 Upload PDFs")
uploaded = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True, key=st.session_state.uploader_key)
scanned_mode = st.sidebar.checkbox("📸 Scanned PDF (image only)?")

if st.sidebar.button("📊 Extract & Index"):
    if uploaded:
        with st.spinner("Processing and indexing..."):
            st.session_state.vs = load_and_index(uploaded, scanned_mode)
            st.success("✅ Documents indexed successfully!")
            st.session_state.uploader_key += 1
        st.session_state.msgs = [{"role": "assistant", "content": "Ask your question about the PDF content or tables!"}]

if st.sidebar.button("🩹 Clear Chat"):
    st.session_state.msgs = []
    st.success("Chat cleared.")

if st.sidebar.button("🗑 Clear DB"):
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    st.session_state.vs = None
    st.success("Database cleared.")

# --- Session State ---
if "vs" not in st.session_state:
    st.session_state.vs = load_existing_index()
if "msgs" not in st.session_state:
    st.session_state.msgs = []

# --- Chat Window ---
for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask your question..."):
    st.session_state.msgs.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    if st.session_state.vs:
        chain = get_chat_chain(st.session_state.vs)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = "".join(chain.stream(query))
                st.markdown(response)
                st.session_state.msgs.append({"role": "assistant", "content": response})
    else:
        st.error("Upload and index documents first.")