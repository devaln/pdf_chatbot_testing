# --- Imports ---
import os, json, uuid, shutil, re
from pathlib import Path
import concurrent.futures
import pytesseract
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path
from difflib import SequenceMatcher
import streamlit as st
import layoutparser as lp
from ChartOCR import ChartOCR
from langchain.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain

# --- Config ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3:latest"
HF_EMBED_MODEL = "intfloat/e5-base"
DB_DIR = "./faiss_db"
TABLE_DIR = "./tables"
TEMP_DIR = "./temp"
TOP_K = 5
TEXT_CHUNK_SIZE = 500

# --- Setup ---
os.makedirs(TABLE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
embedder = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)
chartocr = ChartOCR()
lp_model = lp.Detectron2LayoutModel(
    config_path="lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config",
    extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.85],
    label_map={0: "Text", 1: "Title", 2: "List", 3: "Table", 4: "Figure"}
)

# --- State ---
if "msgs" not in st.session_state:
    st.session_state.msgs = []
if "vs" not in st.session_state:
    st.session_state.vs = None
if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# --- Helpers ---
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def is_similar_header(h1, h2):
    return len(h1) == len(h2) and all(similar(a, b) > 0.7 for a, b in zip(h1, h2))

# --- Extraction Function ---
def extract_layout_elements(pdf_path):
    fname = os.path.basename(pdf_path)
    images = convert_from_path(pdf_path, dpi=300)
    docs = []
    last_headers, last_table_id = None, None

    for pn, img in enumerate(images):
        layout = lp_model.detect(img)
        for block in layout:
            segment = block.crop_image(img)
            block_type = block.type

            if block_type == "Table":
                text = pytesseract.image_to_string(segment)
                lines = [l.strip() for l in text.split("\n") if l.strip()]
                if not lines: continue
                first_row = lines[0].split()
                if all(not c.replace('.', '', 1).isdigit() for c in first_row):
                    headers = first_row
                    rows = [r.split() for r in lines[1:] if len(r.split()) == len(headers)]
                    last_headers = headers
                    last_table_id = str(uuid.uuid4())
                else:
                    headers = last_headers
                    rows = [r.split() for r in lines if headers and len(r.split()) == len(headers)]
                if headers and rows:
                    table_json = {"headers": headers, "rows": rows}
                    json_path = f"{TABLE_DIR}/table_{last_table_id}.json"
                    if os.path.exists(json_path):
                        existing = json.load(open(json_path))
                        existing["rows"].extend(rows)
                        json.dump(existing, open(json_path, "w"))
                    else:
                        json.dump(table_json, open(json_path, "w"))
                    formatted = "\n".join(", ".join(r) for r in rows)
                    text_block = f"Table Chunk\nHeaders: {', '.join(headers)}\nRows:\n{formatted}"
                    docs.append(Document(page_content=text_block, metadata={"source": fname, "page": pn+1, "type": "table"}))

            elif block_type == "Figure":
                try:
                    chart_data = chartocr.predict(segment)
                    if chart_data:
                        headers = chart_data["headers"]
                        rows = chart_data["rows"]
                        cid = str(uuid.uuid4())
                        json.dump(chart_data, open(f"{TABLE_DIR}/chart_{cid}.json", "w"))
                        formatted = "\n".join(", ".join(map(str, r)) for r in rows)
                        chart_text = f"Chart Data\nHeaders: {', '.join(headers)}\nRows:\n{formatted}"
                        docs.append(Document(page_content=chart_text, metadata={"source": fname, "page": pn+1, "type": "chart"}))
                except Exception as e:
                    print(f"ChartOCR failed: {e}")

            elif block_type in ["Title", "Text", "List"]:
                raw = pytesseract.image_to_string(segment).strip()
                if raw:
                    docs.append(Document(page_content=raw, metadata={"source": fname, "page": pn+1, "type": block_type.lower()}))

        fallback = pytesseract.image_to_string(img).strip()
        for i in range(0, len(fallback), TEXT_CHUNK_SIZE):
            chunk = fallback[i:i+TEXT_CHUNK_SIZE].strip()
            if chunk:
                docs.append(Document(page_content=chunk, metadata={"source": fname, "page": pn+1, "type": "text"}))

    return docs

# --- Indexing ---
def process_and_index(files):
    docs = []
    for file in files:
        path = os.path.join(TEMP_DIR, file.name)
        with open(path, "wb") as f:
            f.write(file.getbuffer())
        docs.extend(extract_layout_elements(path))
    if not docs: return None
    if Path(DB_DIR).exists():
        vs = FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
        vs.add_documents(docs)
    else:
        vs = FAISS.from_documents(docs, embedder)
    vs.save_local(DB_DIR)
    shutil.rmtree(TEMP_DIR, ignore_errors=True)
    return vs

# --- Chain ---
def get_conversational_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Cite document name and page. Use prior chat history to understand follow-ups.

Chat History:
{chat_history}

Context from documents:
{context}

Question: {question}
Answer:
""")
    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    memory = st.session_state.memory
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

def smart_split(query):
    return [q.strip() for q in re.split(r'\b(?:and|or|then|next|also|followed by|after that|&|\n)\b', query, flags=re.IGNORECASE) if q.strip()]

def route_query(q, vs):
    if any(w in q.lower() for w in ["sum", "average", "mean", "trend", "total", "compare"]):
        tables = sorted(Path(TABLE_DIR).glob("*.json"), key=os.path.getmtime, reverse=True)
        if not tables:
            return "⚠ No table/chart data available."
        df = pd.read_json(tables[0])
        agent = create_pandas_dataframe_agent(
            ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL),
            df,
            agent_type=AgentType.OPENAI_FUNCTIONS,
            verbose=False
        )
        return agent.run(q)
    else:
        chain = get_conversational_chain(vs)
        return chain.run({"question": q})

# --- Streamlit UI ---
st.set_page_config(page_title="PDF Q&A", layout="wide")
st.title("PDF + Chart Q&A")

with st.sidebar:
    st.header("Upload PDFs")
    uploader = st.file_uploader("Upload scanned PDF(s)", type="pdf", accept_multiple_files=True)
    if uploader:
        st.session_state.uploaded_files = uploader
        for f in uploader:
            st.markdown(f"✅ {f.name}")

    if st.button("Extract & Index"):
        if st.session_state.uploaded_files:
            with st.spinner("Extracting..."):
                vs = process_and_index(st.session_state.uploaded_files)
                if vs:
                    st.session_state.vs = vs
                    st.session_state.msgs = []
                    st.success("Indexing complete!")
                    st.session_state.uploaded_files = []
        else:
            st.warning("Please upload PDFs.")

    if st.button("Clear Chat"): st.session_state.msgs = []
    if st.button("Clear FAISS Index"):
        shutil.rmtree(DB_DIR, ignore_errors=True)
        st.session_state.vs = None
        st.success("Index deleted.")
    if st.button("Clear Memory"):
        st.session_state.memory.clear()
        st.session_state.chat_history = []

if st.session_state.vs is None and Path(DB_DIR).exists():
    st.session_state.vs = FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)

st.markdown("### Ask your question")
query = st.chat_input("Ask questions (e.g. 'total visits and chart trend')")

if query:
    st.session_state.msgs.append({"role": "user", "content": query})
    with st.spinner("Processing..."):
        responses = []
        sub_qs = smart_split(query)
        with concurrent.futures.ThreadPoolExecutor() as ex:
            future_map = {ex.submit(route_query, q, st.session_state.vs): q for q in sub_qs}
            for idx, future in enumerate(concurrent.futures.as_completed(future_map)):
                sub_q = future_map[future]
                try:
                    result = future.result()
                    responses.append(f"### Q{idx+1}: {sub_q}\n\n{result}")

                    # Store Q&A in FAISS
                    qa_doc = Document(
                        page_content=f"Q: {sub_q}\nA: {result}",
                        metadata={"source": "chat_memory", "type": "chat", "user": "default"}
                    )
                    st.session_state.vs.add_documents([qa_doc])
                    st.session_state.vs.save_local(DB_DIR)

                except Exception as e:
                    responses.append(f"### Q{idx+1}: {sub_q}\n\n⚠ Error: {e}")
        st.session_state.msgs.append({"role": "assistant", "content": "\n\n---\n\n".join(responses)})

for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)
