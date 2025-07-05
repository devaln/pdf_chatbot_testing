# --- Imports ---
import os
import json
import uuid
import shutil
import streamlit as st
import pytesseract
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.chat_models import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from pathlib import Path

# --- Config ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3:latest"
HF_EMBED_MODEL = "intfloat/e5-base"
DB_DIR = "./faiss_db"
TOP_K = 5

# --- Embeddings ---
embedder = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)

# --- Streamlit UI ---
st.set_page_config(page_title="PDF QA (Scanned Tables)", layout="wide")
st.title("📄 PDF Q&A (Scanned Tables)")
st.markdown("Ask questions like: 'What is the target for home visits?'")

# --- Session ---
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "vs" not in st.session_state:
    st.session_state.vs = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []

# --- Load FAISS ---
def load_existing_index():
    index_path = Path(DB_DIR) / "index.faiss"
    if not index_path.exists():
        return None
    try:
        return FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
    except Exception as e:
        st.error(f"Failed to load FAISS index: {e}")
        return None

# --- OCR Table Extraction + Chunking ---
def extract_tables_from_pdf(pdf_path):
    all_chunks = []
    os.makedirs("tables", exist_ok=True)

    images = convert_from_path(pdf_path, dpi=300)
    for page_num, img in enumerate(images):
        ocr_df = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
        ocr_df = ocr_df.dropna(subset=["text"])
        ocr_df = ocr_df[ocr_df.text.str.strip() != ""]

        lines = []
        for _, row in ocr_df.iterrows():
            lines.append((row["top"], row["text"]))

        # Group lines by similar 'top' position (i.e., same row)
        grouped = {}
        for top, word in lines:
            key = min(grouped.keys(), default=top + 100)
            found = False
            for k in list(grouped.keys()):
                if abs(k - top) < 10:
                    grouped[k].append(word)
                    found = True
                    break
            if not found:
                grouped[top] = [word]

        # Convert grouped lines into rows
        table_rows = list(grouped.values())

        if len(table_rows) < 2:
            continue  # skip pages without tables

        headers = table_rows[0]
        rows = table_rows[1:]

        table_json = {
            "title": "Operational Achievement",
            "headers": headers,
            "rows": rows
        }

        table_id = str(uuid.uuid4())
        table_path = f"tables/table_{table_id}.json"
        with open(table_path, "w", encoding="utf-8") as f:
            json.dump(table_json, f, indent=2)

        formatted_rows = "\n".join([", ".join(r) for r in rows])
        chunk_text = f"Table Title: Operational Achievement\nHeaders: {', '.join(headers)}\nRows:\n{formatted_rows}"
        all_chunks.append(Document(page_content=chunk_text, metadata={"table_title": "Operational Achievement"}))

    return all_chunks

# --- Index PDFs ---
def process_and_index(files):
    docs = []
    os.makedirs("temp", exist_ok=True)
    for file in files:
        temp_path = os.path.join("temp", file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        docs.extend(extract_tables_from_pdf(temp_path))

    if not docs:
        return None

    if Path(DB_DIR).exists():
        vs = FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
        vs.add_documents(docs)
    else:
        vs = FAISS.from_documents(docs, embedder)

    vs.save_local(DB_DIR)
    return vs

# --- LLM Chain ---
def get_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    prompt = ChatPromptTemplate.from_template(
        """
        You are a smart table extraction assistant.
        - If user asks about a table, return its contents.
        - If user asks about a specific row or column, extract and return only relevant values.
        - Always use only the context provided below.

        Context:
        {context}

        Question: {question}
        Answer:
        """
    )
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# --- Sidebar UI ---
with st.sidebar:
    st.header("📂 Upload Scanned PDFs")
    uploader = st.file_uploader("Upload scanned PDF(s)", type="pdf", accept_multiple_files=True)
    if uploader:
        st.session_state.uploaded_files = uploader
        for f in uploader:
            st.markdown(f"✅ {f.name}")

    if st.button("📄 Extract & Index"):
        if st.session_state.uploaded_files:
            with st.spinner("Extracting tables and indexing..."):
                vs = process_and_index(st.session_state.uploaded_files)
                if vs:
                    st.session_state.vs = vs
                    st.session_state.msgs = []
                    st.success("✅ Indexing complete!")
                    st.session_state.uploaded_files = []
        else:
            st.warning("Please upload PDFs first.")

    if st.button("🧹 Clear Chat"):
        st.session_state.msgs = []

    if st.button("🗑 Clear FAISS Index"):
        shutil.rmtree(DB_DIR, ignore_errors=True)
        st.session_state.vs = None
        st.success("🗑 FAISS index deleted.")

# --- Load existing index if available ---
if st.session_state.vs is None:
    st.session_state.vs = load_existing_index()

# --- Chat UI ---
st.markdown("### 💬 Ask your question")
query = st.chat_input("E.g. 'Target for home visit' or 'Provide operational achievement'")

if query:
    if not st.session_state.vs:
        st.error("Please upload and index PDFs first.")
    else:
        st.session_state.msgs.append({"role": "user", "content": query})
        with st.spinner("Thinking..."):
            try:
                chain = get_chain(st.session_state.vs)
                result = chain.invoke(query)
                if not result or len(result.strip()) < 2:
                    result = "⚠ No relevant data found."
            except Exception as e:
                result = f"⚠ Error: {e}"
            st.session_state.msgs.append({"role": "assistant", "content": result})

# --- Display Chat ---
for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])