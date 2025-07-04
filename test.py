# --- Imports ---
import os
import tempfile
import shutil
import logging
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
import fitz  # PyMuPDF
import pdfplumber
import camelot
import pytesseract
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz
import layoutparser as lp

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

st.set_page_config(page_title="PDF QA with Table Matching + LayoutParser", layout="wide")
st.title("📄 PDF QA with Table Matching + LayoutParser")

# --- Session State Init ---
for key in ["vs", "tables", "msgs", "uploader_key", "debug_images"]:
    if key not in st.session_state:
        st.session_state[key] = [] if key in ["tables", "msgs", "debug_images"] else None
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

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
        row_str = " | ".join(
            f"{(col or '').strip()}: {str(val).strip()}" for col, val in row.items()
        )
        rows.append(row_str)
    return "\n".join(rows)

def extract_tables_pdf(path):
    results = []
    prev_header = None
    current_rows = []
    current_title = None
    with pdfplumber.open(path) as pdf:
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

def extract_tables_layoutparser(pdf_path, show_debug=False):
    model = lp.PaddleDetectionLayoutModel("lp://PubLayNet/ppyolov2_r50vd_dcn_365e_publaynet", enforce_cpu=True)
    images = convert_from_path(pdf_path)
    tables = []
    debug_images = []

    for img in images:
        layout = model.detect(img)
        blocks = [b for b in layout if b.type == 'Text']
        draw_img = img.copy()
        draw = ImageDraw.Draw(draw_img)
        ocr_df = pd.DataFrame(columns=["x1", "y1", "x2", "y2", "text"])
        for block in blocks:
            segment = (int(block.block.x_1), int(block.block.y_1), int(block.block.x_2), int(block.block.y_2))
            cropped = img.crop(segment)
            text = pytesseract.image_to_string(cropped, config="--psm 6").strip()
            if text:
                ocr_df.loc[len(ocr_df)] = [*segment, text]
                draw.rectangle(segment, outline="red", width=2)
                draw.text((segment[0], segment[1]), text[:20], fill="blue")

        if not ocr_df.empty:
            # very basic column inference based on x-coordinates
            ocr_df = ocr_df.sort_values(by=["y1", "x1"])
            grouped_rows = []
            current_y = -100
            row = []
            for _, item in ocr_df.iterrows():
                if abs(item["y1"] - current_y) > 15:
                    if row:
                        grouped_rows.append(row)
                    row = [item["text"]]
                    current_y = item["y1"]
                else:
                    row.append(item["text"])
            if row:
                grouped_rows.append(row)
            max_cols = max(len(r) for r in grouped_rows)
            norm_rows = [r + [""] * (max_cols - len(r)) for r in grouped_rows]
            df = pd.DataFrame(norm_rows)
            df.columns = [f"col{i}" for i in range(df.shape[1])]
            tables.append((df, "Scanned Table"))

        if show_debug:
            debug_images.append(draw_img)

    if show_debug:
        st.session_state.debug_images.extend(debug_images)

    return tables

def extract_all_tables(path, scanned=False, show_debug=False):
    if scanned:
        return extract_tables_layoutparser(path, show_debug)
    return extract_tables_pdf(path)

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
    tool = Tool(name="pandas_agent", description="Run Python to answer using df", func=PythonREPLTool().run)
    prompt = PromptTemplate.from_template("""
You are a pandas expert working with a table loaded as df. Answer with Python.

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

# --- Sidebar ---
st.sidebar.header("📂 Upload PDFs")
uploaded = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True, key=st.session_state.uploader_key)
scanned_mode = st.sidebar.checkbox("📸 Scanned PDF?")
show_debug = st.sidebar.checkbox("🧪 Show OCR Debug View", value=False)

if st.sidebar.button("📊 Extract & Index"):
    if uploaded:
        with st.spinner("Indexing documents..."):
            st.session_state.debug_images = []
            load_and_index(uploaded, scanned=scanned_mode, show_debug=show_debug)
            st.session_state.msgs = [{"role": "assistant", "content": "✅ You can now ask about any table or document!"}]
            st.success("Extraction & Indexing complete ✅")
            st.rerun()

if st.sidebar.button("🧹 Clear Chat"):
    st.session_state.msgs = []
if st.sidebar.button("🗑 Clear DB"):
    shutil.rmtree(DB_DIR, ignore_errors=True)
    st.session_state.vs = None
    st.session_state.tables = []
    st.success("Database cleared.")

# --- Chat UI ---
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
                top_chunks = [
                    f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                    for doc, score in sorted(results, key=lambda x: x[1])[:5]
                ]
                context = "\n\n".join(top_chunks)
                sub_questions = [q.strip() for q in query.replace("&", " and ").split(" and ") if q.strip()]
                responses = []
                llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
                for q in sub_questions:
                    prompt = f"""
Use ONLY the context below to answer. If not found, say "Not found".

Context:
{context}

Question: {q}
Respond in bullet points or structured format.
"""
                    resp = llm.invoke(prompt)
                    responses.append(f"*Q: {q}*\n{resp.strip()}")
                final = "\n\n".join(responses)
                st.markdown(final)
                st.session_state.msgs.append({"role": "assistant", "content": final})
    else:
        st.error("Please upload and index files first.")

# --- Debug Overlay View ---
if st.session_state.debug_images:
    st.subheader("🔍 OCR Debug Overlay")
    for img in st.session_state.debug_images:
        st.image(img, use_column_width=True)