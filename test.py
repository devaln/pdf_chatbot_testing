# --- Imports ---
import os
import tempfile
import shutil
import streamlit as st
import pandas as pd
import numpy as np
import fitz
import pdfplumber
import pytesseract
from pdf2image import convert_from_path
from fuzzywuzzy import fuzz
import layoutparser as lp
from PIL import Image, ImageDraw

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

# --- Session ---
if "vs" not in st.session_state:
    st.session_state.vs = None
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "tables" not in st.session_state:
    st.session_state.tables = []

# --- Helpers ---
def clean_df(df):
    df.columns = pd.io.parsers.ParserBase({'names': df.columns})._maybe_dedup_names(df.columns)
    return df.fillna("")

def df_to_text(df, title=None):
    lines = [f"{title or ''}".strip()]
    for _, row in df.iterrows():
        row_str = " | ".join(f"{col.strip()}: {str(val).strip()}" for col, val in row.items())
        lines.append(row_str)
    return "\n".join(lines)

def extract_tables_layoutparser(pdf_path, show_debug=False):
    model = lp.Detectron2LayoutModel(
        config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
        label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"},
        extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.5],
        enforce_cpu=True,
    )
    pages = convert_from_path(pdf_path)
    all_tables = []

    for i, image in enumerate(pages):
        layout = model.detect(image)
        table_blocks = [b for b in layout if b.type == "Table"]
        if not table_blocks:
            continue

        image_np = np.array(image)
        for j, block in enumerate(table_blocks):
            segment = block.crop_image(image_np)
            ocr_df = pytesseract.image_to_data(segment, output_type=pytesseract.Output.DATAFRAME)
            ocr_df = ocr_df.dropna(subset=["text"])
            ocr_df = ocr_df[ocr_df["conf"].astype(int) > 40]
            grouped = ocr_df.groupby(['block_num', 'par_num', 'line_num'])

            lines = []
            for _, grp in grouped:
                line = grp.sort_values("left")["text"].tolist()
                lines.append(line)

            if not lines:
                continue

            max_cols = max(len(r) for r in lines)
            padded = [r + [""] * (max_cols - len(r)) for r in lines]
            df_raw = pd.DataFrame(padded).replace("", pd.NA).dropna(how="all").fillna("")

            header_idx = df_raw.apply(lambda r: r.str.len().gt(1).sum(), axis=1).idxmax()
            headers = df_raw.iloc[header_idx].tolist()
            headers = [h if h.strip() else f"col{i}" for i, h in enumerate(headers)]
            df = df_raw.iloc[header_idx + 1:].reset_index(drop=True)
            df.columns = headers
            all_tables.append((df, f"Scanned Table Page {i+1}"))

        if show_debug:
            debug_img = image.copy()
            draw = ImageDraw.Draw(debug_img)
            for block in layout:
                x1, y1, x2, y2 = map(int, block.coordinates)
                draw.rectangle([x1, y1, x2, y2], outline="red", width=2)
                draw.text((x1, y1 - 10), block.type, fill="red")
            st.image(debug_img, caption=f"Page {i+1} OCR Debug", use_column_width=True)

    return all_tables

def extract_tables_pdfplumber(pdf_path):
    dfs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            tables = page.extract_tables()
            for t in tables:
                if t and len(t) > 1:
                    df = pd.DataFrame(t[1:], columns=t[0])
                    dfs.append((clean_df(df), ""))
    return dfs

def extract_all_tables(path, scanned=False, show_debug=False):
    if scanned:
        return extract_tables_layoutparser(path, show_debug)
    return extract_tables_pdfplumber(path)

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

def fuzzy_match_table(query):
    best_score = 0
    best_table = None
    for table in st.session_state.tables:
        score = fuzz.partial_ratio(query.lower(), table["table_title"].lower())
        if score > best_score:
            best_score = score
            best_table = table
    return best_table if best_score >= 70 else None

# --- Agent ---
def run_pandas_agent(df, user_query):
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    tool = Tool(name="pandas_agent", description="Analyze a Pandas table", func=PythonREPLTool().run)
    prompt = PromptTemplate.from_template("""You are a Pandas expert. The table is loaded as df.

Question: {input}""")
    agent = create_react_agent(llm, tools=[tool], prompt=prompt)
    executor = AgentExecutor(agent=agent, tools=[tool], verbose=False)

    context = f"import pandas as pd\ndf = pd.DataFrame({df.to_dict(orient='list')})"
    try:
        result = executor.invoke({"input": f"{context}\n\n{user_query}"})
        return result["output"]
    except Exception as e:
        return f"Agent error: {e}"

# --- UI ---
st.sidebar.header("📂 Upload PDFs")
uploaded = st.sidebar.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)
scanned_mode = st.sidebar.checkbox("📸 Is Scanned PDF?", value=False)
show_debug = st.sidebar.checkbox("🛠 Show OCR Debug Overlays", value=False)

if st.sidebar.button("📊 Extract & Index"):
    if uploaded:
        with st.spinner("Indexing..."):
            load_and_index(uploaded, scanned=scanned_mode, show_debug=show_debug)
            st.success("✅ Documents indexed.")
            st.session_state.msgs = [{"role": "assistant", "content": "Ask about any table or row!"}]

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

if query := st.chat_input("Ask about any table or row!"):
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
                top_chunks = [f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
                              for doc, _ in sorted(results, key=lambda x: x[1])]
                context = "\n\n".join(top_chunks)
                llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL)
                sub_qs = [q.strip() for q in query.replace("&", " and ").split(" and ") if q.strip()]
                answers = []
                for sq in sub_qs:
                    prompt = f"""You are an assistant. Use ONLY the context below.

Context:
{context}

Question: {sq}
Answer:"""
                    resp = llm.invoke(prompt).strip()
                    answers.append(f"*Q: {sq}*\n{resp}")
                final = "\n\n".join(answers)
                st.markdown(final)
                st.session_state.msgs.append({"role": "assistant", "content": final})
    else:
        st.error("Please upload and index files first.")