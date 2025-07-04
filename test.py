# --- Imports ---
import os
import tempfile
import shutil
import streamlit as st
import pandas as pd
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from collections import defaultdict

# --- Config ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama4:latest"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DB_DIR = "./faiss_db"

st.set_page_config(page_title="PDF QA using LLaMA4", layout="wide")
st.title("\U0001F4C4 PDF QA using LLaMA4")

# --- Session State ---
if "vs" not in st.session_state:
    st.session_state.vs = None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "tables" not in st.session_state:
    st.session_state.tables = []

# --- Style ---
st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background-color: white;
    border-right: 1px solid #ccc;
}
</style>
""", unsafe_allow_html=True)

# --- Helpers ---
def clean_df(df):
    df.columns = [str(c) if c else f"col{i}" for i, c in enumerate(df.columns)]
    return df.fillna("")

def df_to_text(df, title=None):
    rows = [f"{title or ''}".strip() if title else ""]
    for _, row in df.iterrows():
        row_str = " | ".join(f"{str(col).strip()}: {str(val).strip()}" for col, val in row.items())
        rows.append(row_str)
    return "\n".join(rows)

def extract_scanned_table(pdf_path):
    try:
        images = convert_from_path(pdf_path)
        stitched_tables = []
        prev_headers = None
        current_rows = []

        for img in images:
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
            data = data.dropna(subset=["text"])
            data = data[data['conf'].astype(int) > 40]
            grouped = data.groupby(['block_num', 'par_num', 'line_num'])

            rows = []
            for _, group in grouped:
                line = group.sort_values("left")["text"].tolist()
                rows.append(line)

            if not rows:
                continue

            max_cols = max(len(row) for row in rows)
            padded = [row + [""] * (max_cols - len(row)) for row in rows]
            df_raw = pd.DataFrame(padded).replace("", pd.NA).dropna(how="all").fillna("")

            header_row_idx = df_raw.apply(lambda r: r.str.len().gt(1).sum(), axis=1).idxmax()
            headers = df_raw.iloc[header_row_idx].tolist()
            headers = [h if h.strip() else f"col{i}" for i, h in enumerate(headers)]

            df = df_raw.iloc[header_row_idx + 1:].reset_index(drop=True)
            df.columns = headers

            if prev_headers and headers == prev_headers:
                current_rows.append(df)
            else:
                if current_rows:
                    stitched_tables.append(pd.concat(current_rows, ignore_index=True))
                current_rows = [df]
                prev_headers = headers

        if current_rows:
            stitched_tables.append(pd.concat(current_rows, ignore_index=True))

        return stitched_tables
    except Exception:
        return []

def extract_tables_pdf(pdf_path):
    results = []
    prev_header = None
    current_rows = []
    current_title = None

    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            lines = page.extract_text().split("\n") if page.extract_text() else []
            tables = page.extract_tables()

            for table in tables:
                if table and len(table) > 1:
                    df = clean_df(pd.DataFrame(table[1:], columns=table[0]))
                    header = tuple(df.columns)

                    title = ""
                    for i, line in enumerate(lines):
                        if all(col.strip() in line for col in table[0] if col):
                            if i > 0:
                                title = lines[i - 1].strip()
                            break

                    if prev_header and header == prev_header:
                        current_rows.append(df)
                    else:
                        if current_rows:
                            stitched = pd.concat(current_rows, ignore_index=True)
                            results.append((stitched, current_title or ""))
                        current_rows = [df]
                        prev_header = header
                        current_title = title

        if current_rows:
            stitched = pd.concat(current_rows, ignore_index=True)
            results.append((stitched, current_title or ""))

    return results

def extract_all_tables(path, scanned):
    if scanned:
        dfs = extract_scanned_table(path)
        return [(df, "") for df in dfs]
    return extract_tables_pdf(path)

def load_and_index(files, scanned=False):
    embed = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    docs = []
    tables = []
    with tempfile.TemporaryDirectory() as td:
        for f in files:
            path = os.path.join(td, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())

            table_blocks = extract_all_tables(path, scanned)
            for df, title in table_blocks:
                text = df_to_text(df, title)
                docs.append(Document(page_content=text, metadata={"source": f.name, "table_title": title}))
                tables.append({"data": df, "table_title": title, "source": f.name})

            try:
                doc = fitz.open(path)
                full_text = "\n".join(p.get_text() for p in doc)
                docs.append(Document(page_content=full_text, metadata={"source": f.name}))
            except:
                pass

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    if chunks:
        if st.session_state.vs:
            st.session_state.vs.add_documents(chunks)
        else:
            st.session_state.vs = FAISS.from_documents(chunks, embed)
        st.session_state.vs.save_local(DB_DIR)
    st.session_state.tables.extend(tables)

def split_query(query):
    import re
    query = query.strip()
    questions = re.split(r'\?\s*(and|&|also)?\s*|\s+(and|&|also)\s+', query)
    questions = [q.strip() for q in questions if q and len(q.strip()) > 5]
    return questions

# --- Sidebar UI ---
st.sidebar.header("\U0001F4C2 Upload PDFs")
uploaded = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True, key=st.session_state.uploader_key)
scanned_mode = st.sidebar.checkbox("\U0001F4F8 Is Scanned PDF?", value=False)

if st.sidebar.button("\U0001F4CA Extract & Index"):
    if uploaded:
        with st.spinner("Indexing documents..."):
            load_and_index(uploaded, scanned=scanned_mode)
            st.success("Documents indexed ✅")
            st.session_state.uploader_key += 1
            st.session_state.msgs = [{"role": "assistant", "content": "Ask about any table or row!"}]

if st.sidebar.button("\U0001F9F9 Clear Chat"):
    st.session_state.msgs = []
if st.sidebar.button("\U0001F5D1 Clear DB"):
    shutil.rmtree(DB_DIR, ignore_errors=True)
    st.session_state.vs = None
    st.session_state.tables = []
    st.success("Database cleared.")

# --- Chat Loop ---
for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if query := st.chat_input("Ask a question..."):
    st.session_state.msgs.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    if st.session_state.vs:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                sub_questions = split_query(query)
                all_chunks = []
                numbered_questions = []

                for i, sq in enumerate(sub_questions, 1):
                    results = st.session_state.vs.similarity_search_with_score(sq, k=4)
                    chunks = [doc.page_content for doc, score in results if score < 0.8]
                    all_chunks.extend(chunks)
                    numbered_questions.append(f"Q{i}: {sq}")

                all_chunks = list(dict.fromkeys(all_chunks))  # remove duplicates
                context = "\n\n".join(all_chunks)

                prompt = f"""You are a helpful assistant. Answer each question clearly based only on the context below:

Context:
{context}

Questions:
{chr(10).join(numbered_questions)}

Answer in a numbered format (e.g., A1:, A2:). If you can't find an answer, say 'Not found in documents.'
"""
                llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
                response = llm.invoke(prompt)
                final_answer = response.content if hasattr(response, "content") else str(response)
                st.markdown(final_answer)
                st.session_state.msgs.append({"role": "assistant", "content": final_answer})
    else:
        st.error("Please upload and index files first.")