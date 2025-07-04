# --- Imports ---
import os, tempfile, shutil, logging
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber, camelot, pytesseract
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
OLLAMA_LLM_MODEL = "llama3:latest"  # or "llama4:latest"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DB_DIR = "./faiss_db"

logging.basicConfig(level=logging.INFO, filename="app.log", format="%(asctime)s [%(levelname)s] %(message)s")

st.set_page_config(page_title="PDF QA with Tables", layout="wide")
st.title("📄 PDF Text & Table Extractor + Chat QA")

# --- Helpers ---
def clean_df(df):
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    return df.fillna("")

def headers_similar(h1, h2, threshold=0.8):
    return SequenceMatcher(None, ",".join(h1), ",".join(h2)).ratio() > threshold

def extract_tables_pdfplumber(pdf_path):
    dfs, last_df = [], None
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                for table in page.extract_tables():
                    if table:
                        df = pd.DataFrame(table[1:], columns=table[0])
                        df = clean_df(df)
                        if last_df is not None and headers_similar(last_df.columns.tolist(), df.columns.tolist()):
                            last_df = pd.concat([last_df, df], ignore_index=True)
                        else:
                            if last_df is not None:
                                dfs.append(last_df)
                            last_df = df
            if last_df is not None:
                dfs.append(last_df)
    except Exception as e:
        logging.warning(f"pdfplumber failed: {e}")
    return dfs

def extract_tables_camelot(pdf_path):
    dfs = []
    for flavor in ["lattice", "stream"]:
        try:
            tables = camelot.read_pdf(pdf_path, pages='all', flavor=flavor)
            for t in tables:
                df = t.df
                if df.shape[0] > 1:
                    dfs.append(clean_df(df))
        except Exception as e:
            logging.warning(f"camelot {flavor} failed: {e}")
    return dfs

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

            max_cols = max(len(r) for r in rows)
            padded = [r + [""] * (max_cols - len(r)) for r in rows]
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

        stitched, last_df = [], None
        for df in full_dfs:
            if last_df is not None and headers_similar(last_df.columns.tolist(), df.columns.tolist()):
                last_df = pd.concat([last_df, df], ignore_index=True)
            else:
                if last_df is not None:
                    stitched.append(last_df)
                last_df = df
        if last_df is not None:
            stitched.append(last_df)

        combined = pd.concat(stitched).reset_index(drop=True)
        return combined.to_csv(index=False), combined.to_string(index=False)
    except Exception as e:
        logging.error(f"OCR failed: {e}")
        return "", ""

def extract_all_tables(pdf_path, scanned_mode=False):
    if scanned_mode:
        return extract_scanned_pdf_with_ocr(pdf_path)

    dfs = extract_tables_pdfplumber(pdf_path)
    dfs += extract_tables_camelot(pdf_path)

    try:
        doc = fitz.open(pdf_path)
        full_text = "\n".join([page.get_text() for page in doc])
    except Exception as e:
        logging.error(f"Text extraction failed: {e}")
        full_text = ""

    return "\n\n".join(df.to_csv(index=False) for df in dfs), full_text

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
                tables_text, full_text = extract_all_tables(path, scanned_mode)
                if tables_text.strip() or full_text.strip():
                    all_docs.append(Document(page_content=tables_text + "\n" + full_text, metadata={"source": file.name}))
            except Exception as e:
                st.error(f"❌ {file.name}: Failed to process: {e}")
                logging.error(f"{file.name} failed: {e}")

    if not all_docs:
        return None

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = [chunk for doc in all_docs for chunk in splitter.split_documents([doc])]

    try:
        embed = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
        if os.path.exists(DB_DIR):
            vs = FAISS.load_local(DB_DIR, embed)
            vs.add_documents(chunks)
        else:
            vs = FAISS.from_documents(chunks, embed)
        vs.save_local(DB_DIR)
        return vs
    except Exception as e:
        logging.error(f"Indexing error: {e}")
        return None

# --- Top-K Retriever ---
class TopKFilterRetriever:
    def _init_(self, retriever, top_k=8):
        self.retriever = retriever
        self.top_k = top_k

    def _call_(self, query, config=None):
        docs = self.retriever.get_relevant_documents(query)
        if not docs:
            return ""
        grouped = {}
        for doc in docs:
            grouped.setdefault(doc.metadata.get("source", "unknown"), []).append(doc)
        top_source = max(grouped.items(), key=lambda x: len(x[1]))[0]
        return "\n\n".join([d.page_content for d in grouped[top_source]])

def get_chat_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": 10})
    filtered = TopKFilterRetriever(retriever)
    prompt = ChatPromptTemplate.from_template("You are a table analysis expert.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:")
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    return {"context": filtered, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# --- UI ---
st.sidebar.image("img/ACL_Digital.png", width=180)
st.sidebar.image("img/Cipla_Foundation.png", width=180)
st.sidebar.markdown("<hr>", unsafe_allow_html=True)
st.sidebar.header("📂 Upload PDFs")
uploaded = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
scanned_mode = st.sidebar.checkbox("📸 Scanned PDF (image only)?")

if st.sidebar.button("📊 Extract & Index"):
    if uploaded:
        with st.spinner("Indexing..."):
            vs_result = load_and_index(uploaded, scanned_mode)
            if vs_result:
                st.session_state.vs = vs_result
                st.session_state.msgs = [{"role": "assistant", "content": "✅ You can now ask questions!"}]
                st.success("✅ Documents indexed.")
            else:
                st.error("❌ Indexing failed. No text or tables found.")

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.msgs = []
    st.success("Chat cleared.")

if st.sidebar.button("🗑 Clear DB"):
    if os.path.exists(DB_DIR):
        shutil.rmtree(DB_DIR)
    st.session_state.vs = None
    st.success("Vector store deleted.")

if "vs" not in st.session_state:
    st.session_state.vs = None
if "msgs" not in st.session_state:
    st.session_state.msgs = []

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
            with st.spinner("Answering..."):
                response = "".join(chain.stream(query))
                st.markdown(response)
                st.session_state.msgs.append({"role": "assistant", "content": response})
    else:
        st.error("❌ Upload and index documents first.")