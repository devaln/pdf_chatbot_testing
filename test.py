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
import requests
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz

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

st.set_page_config(page_title="PDF QA", layout="wide")
st.title("📄 PDF QA - Multi-page Tables + Row-Level QA")

# --- Helpers ---
def clean_df(df):
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    return df.fillna("")

def headers_match(df1, df2):
    return df1.shape[1] == df2.shape[1] and all(
        fuzz.ratio(a, b) > 80 for a, b in zip(df1.columns, df2.columns)
    )

def stitch_tables(dfs_with_titles):
    stitched = []
    last_df, last_title = None, None
    for df, title in dfs_with_titles:
        if last_df is not None and headers_match(last_df, df):
            last_df = pd.concat([last_df, df], ignore_index=True)
        else:
            if last_df is not None:
                stitched.append((last_df, last_title))
            last_df, last_title = df, title
    if last_df is not None:
        stitched.append((last_df, last_title))
    return stitched

def extract_tables_pdfplumber(pdf_path):
    dfs = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                lines = page.extract_text().splitlines() if page.extract_text() else []
                tables = page.extract_tables()
                for j, table in enumerate(tables):
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        title = lines[lines.index(table[0][0]) - 1] if table[0][0] in lines else f"PDF Table {i+1}-{j+1}"
                        dfs.append((clean_df(df), title.strip()))
    except Exception as e:
        st.warning(f"pdfplumber failed: {e}")
    return dfs

def extract_tables_camelot(pdf_path):
    dfs = []
    try:
        tables = camelot.read_pdf(pdf_path, pages='all', flavor='stream')
        for i, t in enumerate(tables):
            if t.df.shape[0] > 1 and t.df.shape[1] > 1:
                df = clean_df(t.df)
                title = f"Camelot Table {i+1}"
                dfs.append((df, title))
    except Exception as e:
        st.warning(f"camelot failed: {e}")
    return dfs

def extract_scanned_pdf_with_ocr(pdf_path):
    try:
        images = convert_from_path(pdf_path, dpi=300)
        full_text = ""
        for img in images:
            text = pytesseract.image_to_string(img, lang="eng")
            full_text += text + "\n"

        if not full_text.strip():
            return "", ""

        prompt = f"""Extract all tables from this OCR text and include table titles.
Format:
Table Title: <title>
<CSV>

OCR Text:
{full_text}"""

        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        result = response.json()
        return result.get("response", "").strip(), full_text
    except Exception as e:
        st.error(f"OCR + LLM failed: {e}")
        return "", ""

def extract_all_tables(pdf_path, scanned_mode=False):
    if scanned_mode:
        return extract_scanned_pdf_with_ocr(pdf_path)

    dfs = extract_tables_pdfplumber(pdf_path) + extract_tables_camelot(pdf_path)
    stitched = stitch_tables(dfs)

    try:
        doc = fitz.open(pdf_path)
        text = "\n".join([page.get_text() for page in doc])
    except:
        text = ""

    prompt = f"""Extract all tables and their titles from this document.
Format:
Table Title: <title>
<CSV>

Text:
{text}"""

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={"model": OLLAMA_LLM_MODEL, "prompt": prompt, "stream": False},
            timeout=120
        )
        result = response.json()
        llm_csv = result.get("response", "").strip()
    except:
        llm_csv = ""

    chunks = []
    for df, title in stitched:
        st.subheader(title)
        st.dataframe(df)
        chunk = f"Table Title: {title}\n{df.to_csv(index=False)}"
        chunks.append(chunk)

    if llm_csv:
        chunks.append("LLM-Structured Tables:\n" + llm_csv)

    return "\n\n".join(chunks), text

@st.cache_resource(show_spinner=False)
def load_and_index(files, scanned_mode=False):
    all_docs = []
    with tempfile.TemporaryDirectory() as td:
        for file in files:
            path = os.path.join(td, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())
            try:
                loader = PyPDFLoader(path)
                all_docs.extend(loader.load())
                table_text, raw_text = extract_all_tables(path, scanned_mode)
                all_docs.append(Document(page_content=table_text + "\n" + raw_text, metadata={"source": file.name}))
            except Exception as e:
                st.error(f"Failed to process {file.name}: {e}")

    if not all_docs:
        return None

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(all_docs)
    try:
        embed = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        vs = FAISS.from_documents(chunks, embed)
        vs.save_local(DB_DIR)
        st.success("✅ Indexed successfully.")
        return vs
    except Exception as e:
        st.error(f"FAISS indexing error: {e}")
        return None

def fuzzy_match_table(query, docs):
    best_score = 0
    best = None
    for doc in docs:
        for line in doc.page_content.split("\n")[:3]:
            score = fuzz.partial_ratio(query.lower(), line.lower())
            if score > best_score:
                best_score = score
                best = doc
    return best if best_score > 70 else None

def load_existing_index():
    if not os.path.exists(DB_DIR):
        return None
    try:
        embed = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        return FAISS.load_local(DB_DIR, embed, allow_dangerous_deserialization=True)
    except:
        return None

def get_chat_chain(vs):
    prompt = ChatPromptTemplate.from_template(
        "You are a PDF table analysis expert.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    )
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
    return {"context": vs.as_retriever(), "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

def clear_db():
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)

# --- Sidebar ---
with st.sidebar:
    st.header("📂 Upload PDFs")
    uploaded = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
    scanned_mode = st.checkbox("📸 Is Scanned PDF?")
    run = st.button("📊 Extract & Index")

    st.markdown("---")
    if st.button("🗑 Clear DB"):
        clear_db()
        st.session_state.vs = None
        st.success("Database cleared.")
    if st.button("🧹 Clear Chat"):
        st.session_state.msgs = []
        st.success("Chat cleared.")

# --- State Init ---
if "vs" not in st.session_state:
    st.session_state.vs = load_existing_index()
if "msgs" not in st.session_state:
    st.session_state.msgs = []

# --- Upload Trigger ---
if run and uploaded:
    st.session_state.msgs = []
    with st.spinner("Processing..."):
        st.session_state.vs = load_and_index(uploaded, scanned_mode)
    if st.session_state.vs:
        st.session_state.msgs.append({"role": "assistant", "content": "✅ You can now query tables by name or row values!"})

# --- Chat Loop ---
for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask about the tables or rows..."):
    st.session_state.msgs.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vs:
        chunks = st.session_state.vs.similarity_search(query, k=8)
        table_doc = fuzzy_match_table(query, chunks)

        context = table_doc.page_content if table_doc else "\n\n".join([doc.page_content for doc in chunks[:3]])

        chain = get_chat_chain(st.session_state.vs)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                resp = chain.invoke({"question": query, "context": context})
                st.markdown(resp)
                st.session_state.msgs.append({"role": "assistant", "content": resp})
    else:
        st.error("Please upload and process PDFs first.")