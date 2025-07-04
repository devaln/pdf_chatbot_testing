import os
import tempfile
import shutil
import logging
from collections import defaultdict
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import fitz  # PyMuPDF
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain.tools.python import PythonREPLTool
# from langchain.tools.python.tool import PythonREPLTool
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Config ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama4:latest"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DB_DIR = "./faiss_db"

st.set_page_config(page_title="PDF QA + Table Agent", layout="wide")
st.title("📄 PDF QA with Table Matching + Pandas Agent")

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
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    return df.fillna("")

def df_to_text(df, title=None):
    rows = [f"{title or ''}".strip()]
    for _, row in df.iterrows():
        row_str = " | ".join(f"{col.strip()}: {str(val).strip()}" for col, val in row.items())
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
    except Exception as e:
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
    tables = extract_tables_pdf(path)
    return tables

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

def fuzzy_match_table(query):
    best_score = 0
    best_table = None
    for table in st.session_state.tables:
        score = fuzz.partial_ratio(query.lower(), table["table_title"].lower())
        if score > best_score:
            best_score = score
            best_table = table
    return best_table if best_score >= 70 else None

def run_pandas_agent(df, user_query):
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    tool = Tool(name="pandas_agent", description="Execute Python code to analyze a table", func=PythonREPLTool().run)
    prompt = PromptTemplate.from_template("""
You are a pandas expert working with a table loaded as df.
Answer the user's question using Python.

Question: {input}
""")
    agent = create_react_agent(llm, tools=[tool], prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=[tool], verbose=False)
    context = f"import pandas as pd\ndf = pd.DataFrame({df.to_dict(orient='list')})"
    try:
        result = executor.invoke({"input": f"{context}\n\n{user_query}"})
        return result["output"]
    except Exception as e:
        return f"Agent error: {e}"

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

    table = fuzzy_match_table(query)
    if table:
        with st.chat_message("assistant"):
            with st.spinner(f"Using table: {table['table_title']}"):
                result = run_pandas_agent(table["data"], query)
                st.markdown(f"*Matched Table:* {table['table_title']}\n\n{result}")
                st.session_state.msgs.append({"role": "assistant", "content": result})
    elif st.session_state.vs:
        with st.chat_message("assistant"):
            with st.spinner("Searching documents..."):
                results = st.session_state.vs.similarity_search_with_score(query, k=6)
                doc_chunks = defaultdict(list)
                doc_scores = defaultdict(list)
                for doc, score in results:
                    doc_chunks[doc.metadata.get("source", "")].append(doc.page_content)
                    doc_scores[doc.metadata.get("source", "")].append(score)
                best_doc = min(doc_scores, key=lambda d: np.mean(doc_scores[d]))
                context = "\n\n".join(doc_chunks[best_doc])
                prompt = f"You are an expert. Answer strictly using the below context:\n\n{context}\n\nQuestion: {query}"
                llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
                response = llm.invoke(prompt)
                st.markdown(response)
                st.session_state.msgs.append({"role": "assistant", "content": response})
    else:
        st.error("Please upload and index files first.")