# --- Imports ---
import os
import shutil
import tempfile
import logging
from collections import defaultdict

import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import fitz  # <-- make sure this is from PyMuPDF
import pytesseract
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz
import matplotlib.pyplot as plt

from langchain.agents import Tool, AgentExecutor, create_react_agent
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import PromptTemplate
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- Config ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3"
OLLAMA_EMBEDDING_MODEL = "nomic-embed-text"
DB_DIR = "./faiss_db"

st.set_page_config(page_title="PDF QA with Table Matching", layout="wide")
st.title("📄 PDF QA with Table Matching")

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
def df_to_text(df, title=None):
    lines = [f"{title or ''}".strip()] if title else []
    for _, row in df.iterrows():
        row_str = " | ".join(f"{(col or '').strip()}: {str(val).strip()}" for col, val in row.items())
        lines.append(row_str)
    return "\n".join(lines)

def extract_tables_scanned_ocr(pdf_path, show_debug=False):
    try:
        images = convert_from_path(pdf_path)
        all_tables = []

        for page_num, image in enumerate(images):
            ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME)
            ocr_data = ocr_data[ocr_data.conf != -1].dropna(subset=['text'])
            ocr_data = ocr_data[ocr_data.text.str.strip().astype(bool)]

            lines = []
            current_line = []
            last_y = None

            for _, row in ocr_data.iterrows():
                if last_y is None or abs(row['top'] - last_y) <= 10:
                    current_line.append(row)
                else:
                    lines.append(current_line)
                    current_line = [row]
                last_y = row['top']
            if current_line:
                lines.append(current_line)

            table_rows = []
            for line in lines:
                sorted_line = sorted(line, key=lambda r: r['left'])
                table_rows.append([r['text'] for r in sorted_line])

            if not table_rows:
                continue

            max_cols = max(len(r) for r in table_rows)
            padded = [r + [""] * (max_cols - len(r)) for r in table_rows]
            df = pd.DataFrame(padded)

            header_idx = df.apply(lambda r: r.str.len().gt(2).sum(), axis=1).idxmax()
            headers = df.iloc[header_idx].tolist()
            headers = [h if h.strip() else f"col{i}" for i, h in enumerate(headers)]
            df_data = df.iloc[header_idx + 1:].reset_index(drop=True)
            df_data.columns = headers
            all_tables.append(df_data)

            if show_debug:
                fig, ax = plt.subplots()
                ax.imshow(image)
                for _, row in ocr_data.iterrows():
                    x, y, w, h = row['left'], row['top'], row['width'], row['height']
                    ax.add_patch(plt.Rectangle((x, y), w, h, edgecolor='red', fill=False, linewidth=1))
                    ax.text(x, y - 2, row['text'], fontsize=5, color='blue')
                ax.set_title(f"OCR Overlay - Page {page_num+1}")
                st.pyplot(fig)

        return [(df, "") for df in all_tables]
    except Exception as e:
        st.error(f"OCR table extraction failed: {e}")
        return []

def extract_tables_text(pdf_path):
    results = []
    try:
        doc = fitz.open(pdf_path)  # ✅ Will now work after uninstalling wrong fitz
        full_text = "\n".join([page.get_text() for page in doc])
        results.append(Document(page_content=full_text, metadata={"source": os.path.basename(pdf_path)}))
    except Exception as e:
        st.warning(f"Failed to extract text: {e}")
    return results

def extract_all_tables(path, scanned=False, show_debug=False):
    if scanned:
        return extract_tables_scanned_ocr(path, show_debug)
    return []

def load_and_index(files, scanned=False, show_debug=False):
    embed = OllamaEmbeddings(model=OLLAMA_EMBEDDING_MODEL, base_url=OLLAMA_BASE_URL)
    docs = []
    tables = []

    with tempfile.TemporaryDirectory() as td:
        for f in files:
            path = os.path.join(td, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())

            table_blocks = extract_all_tables(path, scanned, show_debug)
            for df, title in table_blocks:
                text = df_to_text(df, title)
                docs.append(Document(page_content=text, metadata={"source": f.name, "table_title": title}))
                tables.append({"data": df, "table_title": title, "source": f.name})

            docs += extract_tables_text(path)

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
    tool = Tool(name="pandas_agent", description="Executes Python code to analyze a table", func=PythonREPLTool().run)
    prompt = PromptTemplate.from_template("""
You are a pandas expert working with a DataFrame df.
Answer the user's question using Python code only.

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
st.sidebar.header("📂 Upload PDFs")
uploaded = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True, key=st.session_state.uploader_key)
scanned_mode = st.sidebar.checkbox("📸 Is Scanned PDF?", value=False)
show_debug = st.sidebar.checkbox("🪟 Show OCR Debug View", value=False)

if st.sidebar.button("📊 Extract & Index"):
    if uploaded:
        with st.spinner("Indexing documents..."):
            load_and_index(uploaded, scanned=scanned_mode, show_debug=show_debug)
            st.success("✅ Documents indexed!")
            st.session_state.uploader_key += 1
            st.session_state.msgs = [{"role": "assistant", "content": "✅ Ask about any table or row!"}]

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.msgs = []

if st.sidebar.button("🗑 Clear DB"):
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
                top_chunks = [f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}" for doc, _ in results[:3]]
                context = "\n\n".join(top_chunks)

                llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
                prompt = f"""
You are an expert assistant. Use ONLY the context below to answer the question.
If the answer is not present, say "Not found in the provided documents."

Context:
{context}

Question: {query}
Answer in bullet points or structured format.
"""
                response = llm.invoke(prompt)
                response_text = response.content.strip() if hasattr(response, "content") else str(response).strip()

                st.markdown(response_text)
                st.session_state.msgs.append({"role": "assistant", "content": response_text})
    else:
        st.error("⚠ Please upload and index files first.")