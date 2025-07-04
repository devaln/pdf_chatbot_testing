# --- Imports ---
import os
import tempfile
import shutil
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
import camelot
import pytesseract
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz
import layoutparser as lp

from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_core.prompts import PromptTemplate

# --- Config ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DB_DIR = "./faiss_db"

st.set_page_config(page_title="PDF QA + LayoutParser", layout="wide")
st.title("📄 PDF QA with Table Matching + LayoutParser")

# --- Session State ---
if "vs" not in st.session_state:
    st.session_state.vs = None
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "tables" not in st.session_state:
    st.session_state.tables = []

# --- Helpers ---
def dedup_columns(columns):
    seen = {}
    result = []
    for col in columns:
        count = seen.get(col, 0)
        new_col = f"{col}_{count}" if count else col
        result.append(new_col)
        seen[col] = count + 1
    return result

def clean_df(df):
    df.columns = dedup_columns(df.columns)
    return df.fillna("")

def df_to_text(df, title=None):
    rows = [f"{title or ''}".strip()]
    for _, row in df.iterrows():
        row_str = " | ".join(f"{(col or '').strip()}: {str(val).strip()}" for col, val in row.items())
        rows.append(row_str)
    return "\n".join(rows)
# --- Scanned PDF Table Extraction using LayoutParser + Tesseract ---
def extract_tables_layoutparser(pdf_path, show_debug=False):
    all_dfs = []
    images = convert_from_path(pdf_path)

    model = lp.TesseractLayoutModel(lang='eng')
    
    for i, img in enumerate(images):
        layout = model.detect(img)
        blocks = [b for b in layout if b.type == 'Text']

        if show_debug:
            draw = lp.draw_box(img, layout, box_width=1)
            st.image(draw, caption=f"OCR Blocks - Page {i+1}", use_column_width=True)

        rows = []
        for b in sorted(blocks, key=lambda b: (b.block.y_1, b.block.x_1)):
            text = b.text.strip()
            if text:
                rows.append((int(b.block.y_1), text))

        clustered = {}
        for y, text in rows:
            key = y // 10  # group y-coordinates
            clustered.setdefault(key, []).append(text)

        table_data = list(clustered.values())
        max_len = max(len(row) for row in table_data)
        table_data = [row + [""] * (max_len - len(row)) for row in table_data]

        df_raw = pd.DataFrame(table_data).replace("", pd.NA).dropna(how="all").fillna("")
        if df_raw.empty: continue

        header_row_idx = df_raw.apply(lambda r: r.str.len().gt(1).sum(), axis=1).idxmax()
        headers = df_raw.iloc[header_row_idx].tolist()
        headers = [h if h.strip() else f"col{i}" for i, h in enumerate(headers)]

        df = df_raw.iloc[header_row_idx + 1:].reset_index(drop=True)
        df.columns = dedup_columns(headers)

        if not df.empty:
            all_dfs.append((df, f"Scanned Page {i+1}"))

    return all_dfs

# --- Unscanned Table Extraction ---
def extract_tables_pdf(pdf_path):
    dfs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for table in tables:
                if table and len(table) > 1:
                    df = clean_df(pd.DataFrame(table[1:], columns=table[0]))
                    title = ""
                    lines = page.extract_text().split("\n") if page.extract_text() else []
                    for i, line in enumerate(lines):
                        if all(col.strip() in line for col in table[0] if col):
                            if i > 0: title = lines[i - 1].strip()
                            break
                    dfs.append((df, title))
    return dfs

# --- Main Table Extractor ---
def extract_all_tables(pdf_path, scanned=False, show_debug=False):
    if scanned:
        return extract_tables_layoutparser(pdf_path, show_debug)
    return extract_tables_pdf(pdf_path)

# --- Indexing ---
def load_and_index(files, scanned=False, show_debug=False):
    embed = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    docs, tables = [], []

    with tempfile.TemporaryDirectory() as td:
        for file in files:
            path = os.path.join(td, file.name)
            with open(path, "wb") as f:
                f.write(file.getbuffer())

            table_blocks = extract_all_tables(path, scanned, show_debug)
            for df, title in table_blocks:
                text = df_to_text(df, title)
                docs.append(Document(page_content=text, metadata={"source": file.name, "table_title": title}))
                tables.append({"data": df, "table_title": title, "source": file.name})

            try:
                doc = fitz.open(path)
                full_text = "\n".join(p.get_text() for p in doc)
                docs.append(Document(page_content=full_text, metadata={"source": file.name}))
            except Exception as e:
                st.warning(f"Text extraction failed: {e}")

    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
    if chunks:
        if st.session_state.vs:
            st.session_state.vs.add_documents(chunks)
        else:
            st.session_state.vs = FAISS.from_documents(chunks, embed)
        st.session_state.vs.save_local(DB_DIR)

    st.session_state.tables.extend(tables)
    st.success("✅ Indexing complete!")
# --- UI Sidebar ---
st.sidebar.header("📂 Upload PDFs")
uploaded = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True, key=st.session_state.uploader_key)
scanned_mode = st.sidebar.checkbox("📸 Is Scanned PDF?", value=False)
show_debug = st.sidebar.checkbox("🧪 Show OCR Debug Overlays", value=False)

if st.sidebar.button("📊 Extract & Index"):
    if uploaded:
        with st.spinner("🔍 Extracting tables and building index..."):
            load_and_index(uploaded, scanned=scanned_mode, show_debug=show_debug)
            st.session_state.uploader_key += 1
            st.session_state.msgs = [{"role": "assistant", "content": "Ask about any table or row!"}]

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.msgs = []
if st.sidebar.button("🗑 Clear DB"):
    shutil.rmtree(DB_DIR, ignore_errors=True)
    st.session_state.vs = None
    st.session_state.tables = []
    st.success("Database cleared.")

# --- Display Chat Messages ---
for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- Fuzzy Table Matching Logic ---
def fuzzy_match_table(query):
    best_score, best_table = 0, None
    for table in st.session_state.tables:
        score = fuzz.partial_ratio(query.lower(), table["table_title"].lower())
        if score > best_score:
            best_score = score
            best_table = table
    return best_table if best_score >= 70 else None

# --- LLM Agent with Pandas ---
def run_pandas_agent(df, user_query):
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    tool = Tool(name="pandas_agent", description="Use Python to analyze the table", func=PythonREPLTool().run)
    prompt = PromptTemplate.from_template("""You are a pandas expert with a DataFrame df. Use code to answer:\n\nQuestion: {input}""")
    agent = create_react_agent(llm, tools=[tool], prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=[tool], verbose=False)

    context = f"import pandas as pd\ndf = pd.DataFrame({df.to_dict(orient='list')})"
    try:
        result = executor.invoke({"input": f"{context}\n\n{user_query}"})
        return result["output"]
    except Exception as e:
        return f"Agent error: {e}"

# --- Chat Handler ---
if query := st.chat_input("Ask a question..."):
    st.session_state.msgs.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    matched_table = fuzzy_match_table(query)
    if matched_table:
        with st.chat_message("assistant"):
            with st.spinner(f"Using table: {matched_table['table_title']}"):
                response = run_pandas_agent(matched_table["data"], query)
                st.markdown(f"*Matched Table:* {matched_table['table_title']}\n\n{response}")
                st.session_state.msgs.append({"role": "assistant", "content": response})
    elif st.session_state.vs:
        with st.chat_message("assistant"):
            with st.spinner("Searching all documents..."):
                results = st.session_state.vs.similarity_search_with_score(query, k=6)
                top_chunks = [
                    f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                    for doc, score in sorted(results, key=lambda x: x[1])[:5]
                ]
                context = "\n\n".join(top_chunks)
                sub_questions = [q.strip() for q in query.replace("&", " and ").split(" and ") if q.strip()]
                responses = []
                llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
                for sq in sub_questions:
                    prompt = f"""Use ONLY the context below to answer:\n\nContext:\n{context}\n\nQuestion: {sq}\n\nAnswer:"""
                    resp = llm.invoke(prompt)
                    responses.append(f"*Q: {sq}*\n{resp.strip()}")
                final = "\n\n".join(responses)
                st.markdown(final)
                st.session_state.msgs.append({"role": "assistant", "content": final})
    else:
        st.error("Please upload and index files first.")