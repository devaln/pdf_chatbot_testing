# # --- Imports ---
# import os
# import json
# import uuid
# import shutil
# import streamlit as st
# import pytesseract
# import pandas as pd
# from PIL import Image
# from pdf2image import convert_from_path
# from langchain.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_community.chat_models import ChatOllama
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from pathlib import Path
# import re

# # --- Config ---
# OLLAMA_BASE_URL = "http://localhost:11434"
# OLLAMA_LLM_MODEL = "llama3:latest"
# HF_EMBED_MODEL = "intfloat/e5-base"
# DB_DIR = "./faiss_db"
# TOP_K = 5
# ROW_CHUNK_SIZE = 10
# TEXT_CHUNK_SIZE = 500

# # --- Embeddings ---
# embedder = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)

# # --- Streamlit UI ---
# st.set_page_config(page_title="PDF QA", layout="wide")
# st.title("Upload your PDF Documents or Scanned Documents in .pdf format.)")
# # st.markdown("Ask questions like: 'What is the target for home visits?' or combine multiple queries with 'and', 'then'...")

# # --- Session ---
# if "msgs" not in st.session_state:
#     st.session_state.msgs = []
# if "vs" not in st.session_state:
#     st.session_state.vs = None
# if "uploaded_files" not in st.session_state:
#     st.session_state.uploaded_files = []

# # --- Load FAISS ---
# def load_existing_index():
#     index_path = Path(DB_DIR) / "index.faiss"
#     if not index_path.exists():
#         return None
#     try:
#         return FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
#     except Exception as e:
#         st.error(f"Failed to load FAISS index: {e}")
#         return None

# # --- OCR Table and Text Extraction ---
# def extract_tables_and_text_from_pdf(pdf_path, file_name):
#     all_chunks = []
#     os.makedirs("tables", exist_ok=True)

    
#     print(file_name)
    

#     images = convert_from_path(pdf_path, dpi=300)
#     for page_num, img in enumerate(images):
#         # --- TEXT CHUNKS ---
#         full_text = pytesseract.image_to_string(img)
#         for i in range(0, len(full_text), TEXT_CHUNK_SIZE):
#             text_chunk = full_text[i:i+TEXT_CHUNK_SIZE].strip()
#             if text_chunk:
#                 all_chunks.append(Document(
#                     page_content=f"{text_chunk}",
#                     metadata={
#                         "source": file_name,
#                         "page": page_num,
#                         "type": "text"
#                     }
#                 ))

#         # --- TABLE CHUNKS ---
#         ocr_df = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
#         if "text" not in ocr_df.columns:
#             continue

#         ocr_df["text"] = ocr_df["text"].astype(str)
#         ocr_df = ocr_df[ocr_df["text"].str.strip() != ""]

#         lines = []
#         for _, row in ocr_df.iterrows():
#             try:
#                 lines.append((int(row["top"]), row["text"]))
#             except:
#                 continue

#         grouped = {}
#         for top, word in lines:
#             found = False
#             for k in grouped.keys():
#                 if abs(k - top) < 10:
#                     grouped[k].append(word)
#                     found = True
#                     break
#             if not found:
#                 grouped[top] = [word]

#         table_rows = list(grouped.values())
#         if len(table_rows) < 2:
#             continue

#         headers = table_rows[0]
#         rows = table_rows[1:]
#         if not headers or not rows:
#             continue

#         clean_rows = [r for r in rows if len(r) == len(headers)]
#         if not clean_rows:
#             continue

#         for i in range(0, len(clean_rows), ROW_CHUNK_SIZE):
#             partial_rows = clean_rows[i:i+ROW_CHUNK_SIZE]
#             formatted_rows = "\n".join([", ".join(r) for r in partial_rows])
#             chunk_text = f"Page {page_num+1} Table Chunk\nHeaders: {', '.join(headers)}\nRows:\n{formatted_rows}"
#             all_chunks.append(Document(
#                     page_content=f"{chunk_text}",
#                     metadata={
#                         "source": file_name,
#                         "page": page_num,
#                         "type": "text"
#                     }
#                 ))

#         # Save table JSON (optional)
#         table_json = {
#             "headers": headers,
#             "rows": clean_rows
#         }
#         table_id = str(uuid.uuid4())
#         table_path = f"tables/table_{table_id}.json"
#         with open(table_path, "w", encoding="utf-8") as f:
#             json.dump(table_json, f, indent=2)

#     return all_chunks

# # --- Index PDFs ---
# def process_and_index(files):
#     docs = []
#     os.makedirs("temp", exist_ok=True)
#     for file in files:
#         temp_path = os.path.join("temp", file.name)
#         with open(temp_path, "wb") as f:
#             f.write(file.getbuffer())
#         docs.extend(extract_tables_and_text_from_pdf(temp_path, file.name))

#     if not docs:
#         return None

#     if Path(DB_DIR).exists():
#         vs = FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
#         vs.add_documents(docs)
#     else:
#         vs = FAISS.from_documents(docs, embedder)

#     vs.save_local(DB_DIR)
#     shutil.rmtree("temp", ignore_errors=True)
#     return vs

# # --- LLM Chain ---
# def get_chain(vs):
#     retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
#     prompt = ChatPromptTemplate.from_template(
#         """
#         You are a smart document assistant.
#         - Answer questions based on extracted text and tables from uploaded PDFs.
#         - For every fact you mention, include the document name and page number from which it was retrieved (e.g., *[source.pdf, page 3]*).
#         - Use bullet points, tables, or markdown formatting when possible.
#         - Use clear bullet points, tables, or markdown formatting whenever appropriate.
#         - If the answer cannot be found in the provided context, respond explicitly that the information is not available.
#         - Always mention the document name and page number(s) in your answer if available.
#         - Do not include document IDs or internal system metadata in your response.

#         Context (includes document name and page information):
#         {context}

#         Question:
#         {question}

#         Answer:
#         """
#     )
#     llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)

#     print(f"****************************")
#     print(prompt)
#     print(f"****************************")

#     return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# # --- Sidebar UI ---
# with st.sidebar:
#     st.image("img/ACL_Digital.png", width=180)
#     st.image("img/Cipla_Foundation.png", width=180)
#     st.markdown("---")
#     st.header("Upload Scanned PDFs")
#     uploader = st.file_uploader("Upload scanned PDF(s)", type="pdf", accept_multiple_files=True)
#     if uploader:
#         st.session_state.uploaded_files = uploader
#         for f in uploader:
#             st.markdown(f"\u2705 {f.name}")

#     if st.button("Extract & Index"):
#         if st.session_state.uploaded_files:
#             with st.spinner("Extracting tables and text..."):
#                 vs = process_and_index(st.session_state.uploaded_files)
#                 if vs:
#                     st.session_state.vs = vs
#                     st.session_state.msgs = []
#                     st.success(" Indexing complete!")
#                     st.session_state.uploaded_files = []
#         else:
#             st.warning("Please upload PDFs first.")

#     if st.button("Clear Chat"):
#         st.session_state.msgs = []

#     if st.button("Clear FAISS Index"):
#         shutil.rmtree(DB_DIR, ignore_errors=True)
#         st.session_state.vs = None
#         st.success(" FAISS index deleted.")

# # --- Load existing index if available ---
# if st.session_state.vs is None:
#     st.session_state.vs = load_existing_index()

# # --- Smart Split ---
# def smart_split(query):
#     return [q.strip() for q in re.split(r'\b(?:and|or|then|next|also|after that|followed by|&|\n)\b', query, flags=re.IGNORECASE) if q.strip()]

# # --- Chat UI ---
# st.markdown("### Ask your question")
# query = st.chat_input("Ask questions (e.g. 'target and number trained')")

# if query:
#     if not st.session_state.vs:
#         st.error("Please upload and index PDFs first.")
#     else:
#         sub_queries = smart_split(query)
#         st.session_state.msgs.append({"role": "user", "content": query})

#         with st.spinner("Thinking through your multi-part query..."):
#             try:
#                 chain = get_chain(st.session_state.vs)
#                 responses = []

#                 for idx, sub_q in enumerate(sub_queries):
#                     try:
#                         result = chain.invoke(sub_q)
#                         if result and result.strip():
#                             responses.append(f"### Q{idx+1}: {sub_q}\n\n{result.strip()}")
#                         else:
#                             responses.append(f"### Q{idx+1}: {sub_q}\n\n⚠ No relevant data found.")
#                     except Exception as sub_e:
#                         responses.append(f"### Q{idx+1}: {sub_q}\n\n⚠ Error: {sub_e}")

#                 final_result = "\n\n---\n\n".join(responses)
#             except Exception as e:
#                 final_result = f"⚠ Critical Error: {e}"

#             st.session_state.msgs.append({"role": "assistant", "content": final_result})

# # --- Display Chat ---
# for msg in st.session_state.msgs:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"], unsafe_allow_html=True)









# # --- Imports --- *********************************************88
# import os
# import json
# import uuid
# import shutil
# import streamlit as st
# import pytesseract
# import pandas as pd
# from PIL import Image
# from pdf2image import convert_from_path
# from langchain.vectorstores import FAISS
# from langchain_core.documents import Document
# from langchain_community.chat_models import ChatOllama
# from langchain_community.embeddings import HuggingFaceEmbeddings
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnablePassthrough
# from pathlib import Path
# import re

# # --- Config ---
# OLLAMA_BASE_URL = "http://localhost:11434"
# OLLAMA_LLM_MODEL = "llama3:latest"
# HF_EMBED_MODEL = "intfloat/e5-base"
# DB_DIR = "./faiss_db"
# TOP_K = 5
# ROW_CHUNK_SIZE = 10
# TEXT_CHUNK_SIZE = 500

# # --- Embeddings ---
# embedder = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)

# # --- Streamlit UI ---
# st.set_page_config(page_title="PDF QA", layout="wide")
# st.title("PDF Q&A (Scanned Tables)")
# st.markdown("Ask questions like: 'What is the target for home visits?' or combine multiple queries with 'and', 'then'...")

# # --- Session ---
# if "msgs" not in st.session_state:
#     st.session_state.msgs = []
# if "vs" not in st.session_state:
#     st.session_state.vs = None
# if "uploaded_files" not in st.session_state:
#     st.session_state.uploaded_files = []

# # --- Load FAISS ---
# def load_existing_index():
#     index_path = Path(DB_DIR) / "index.faiss"
#     if not index_path.exists():
#         return None
#     try:
#         return FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
#     except Exception as e:
#         st.error(f"Failed to load FAISS index: {e}")
#         return None

# # --- OCR Table and Text Extraction ---
# def extract_tables_and_text_from_pdf(pdf_path):
#     all_chunks = []
#     os.makedirs("tables", exist_ok=True)

#     filename = os.path.basename(pdf_path)
#     images = convert_from_path(pdf_path, dpi=300)

#     for page_num, img in enumerate(images):
#         # --- TEXT CHUNKS ---
#         full_text = pytesseract.image_to_string(img)
#         for i in range(0, len(full_text), TEXT_CHUNK_SIZE):
#             text_chunk = full_text[i:i+TEXT_CHUNK_SIZE].strip()
#             if text_chunk:
#                 all_chunks.append(Document(
#                     page_content=f"[Document: {filename} | Page: {page_num+1}]\n{text_chunk}",
#                     metadata={"source": filename, "page": page_num + 1}
#                 ))

#         # --- TABLE CHUNKS ---
#         ocr_df = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
#         if "text" not in ocr_df.columns:
#             continue

#         ocr_df["text"] = ocr_df["text"].astype(str)
#         ocr_df = ocr_df[ocr_df["text"].str.strip() != ""]

#         lines = []
#         for _, row in ocr_df.iterrows():
#             try:
#                 lines.append((int(row["top"]), row["text"]))
#             except:
#                 continue

#         grouped = {}
#         for top, word in lines:
#             found = False
#             for k in grouped.keys():
#                 if abs(k - top) < 10:
#                     grouped[k].append(word)
#                     found = True
#                     break
#             if not found:
#                 grouped[top] = [word]

#         table_rows = list(grouped.values())
#         if len(table_rows) < 2:
#             continue

#         headers = table_rows[0]
#         rows = table_rows[1:]
#         if not headers or not rows:
#             continue

#         clean_rows = [r for r in rows if len(r) == len(headers)]
#         if not clean_rows:
#             continue

#         for i in range(0, len(clean_rows), ROW_CHUNK_SIZE):
#             partial_rows = clean_rows[i:i+ROW_CHUNK_SIZE]
#             formatted_rows = "\n".join([", ".join(r) for r in partial_rows])
#             chunk_text = f"[Document: {filename} | Page: {page_num+1}] Table Chunk\nHeaders: {', '.join(headers)}\nRows:\n{formatted_rows}"

#             all_chunks.append(Document(
#                 page_content=chunk_text,
#                 metadata={"source": filename, "page": page_num + 1}
#             ))

#         # Save table JSON
#         table_json = {"headers": headers, "rows": clean_rows}
#         table_id = str(uuid.uuid4())
#         table_path = f"tables/table_{table_id}.json"
#         with open(table_path, "w", encoding="utf-8") as f:
#             json.dump(table_json, f, indent=2)

#     return all_chunks

# # --- Index PDFs ---
# def process_and_index(files):
#     docs = []
#     os.makedirs("temp", exist_ok=True)
#     for file in files:
#         temp_path = os.path.join("temp", file.name)
#         with open(temp_path, "wb") as f:
#             f.write(file.getbuffer())
#         docs.extend(extract_tables_and_text_from_pdf(temp_path))

#     if not docs:
#         return None

#     if Path(DB_DIR).exists():
#         vs = FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
#         vs.add_documents(docs)
#     else:
#         vs = FAISS.from_documents(docs, embedder)

#     vs.save_local(DB_DIR)
#     shutil.rmtree("temp", ignore_errors=True)
#     return vs

# # --- LLM Chain ---
# def get_chain(vs):
#     retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
#     prompt = ChatPromptTemplate.from_template(
#         """
#         You are a smart document assistant.
#         - Answer questions based on extracted text and tables.
#         - Use bullet points, tables, or markdown formatting when possible.
#         - If the answer is not found, respond clearly.
#         - Mention the document name and page number in your answer if available.
#         - Do not include document ID or internal metadata.

#         Context (includes document name and page if present):
#         {context}

#         Question: {question}
#         Answer:
#         """
#     )
#     llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
#     return {"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# # --- Sidebar UI ---
# with st.sidebar:
#     st.image("img/ACL_Digital.png", width=180)
#     st.image("img/Cipla_Foundation.png", width=180)
#     st.markdown("---")
#     st.header("Upload Scanned PDFs")
#     uploader = st.file_uploader("Upload scanned PDF(s)", type="pdf", accept_multiple_files=True)
#     if uploader:
#         st.session_state.uploaded_files = uploader
#         for f in uploader:
#             st.markdown(f"\u2705 {f.name}")

#     if st.button("Extract & Index"):
#         if st.session_state.uploaded_files:
#             with st.spinner("Extracting tables and text..."):
#                 vs = process_and_index(st.session_state.uploaded_files)
#                 if vs:
#                     st.session_state.vs = vs
#                     st.session_state.msgs = []
#                     st.success("Indexing complete!")
#                     st.session_state.uploaded_files = []
#         else:
#             st.warning("Please upload PDFs first.")

#     if st.button("Clear Chat"):
#         st.session_state.msgs = []

#     if st.button("Clear FAISS Index"):
#         shutil.rmtree(DB_DIR, ignore_errors=True)
#         st.session_state.vs = None
#         st.success("FAISS index deleted.")

# # --- Load existing index if available ---
# if st.session_state.vs is None:
#     st.session_state.vs = load_existing_index()

# # --- Smart Split ---
# def smart_split(query):
#     return [q.strip() for q in re.split(r'\b(?:and|or|then|next|also|after that|followed by|&|\n)\b', query, flags=re.IGNORECASE) if q.strip()]

# # --- Chat UI ---
# st.markdown("### Ask your question")
# query = st.chat_input("Ask questions (e.g. 'target and number trained')")

# if query:
#     if not st.session_state.vs:
#         st.error("Please upload and index PDFs first.")
#     else:
#         sub_queries = smart_split(query)
#         st.session_state.msgs.append({"role": "user", "content": query})

#         with st.spinner("Thinking through your multi-part query..."):
#             try:
#                 chain = get_chain(st.session_state.vs)
#                 responses = []

#                 for idx, sub_q in enumerate(sub_queries):
#                     try:
#                         result = chain.invoke(sub_q)
#                         if result and result.strip():
#                             responses.append(f"### Q{idx+1}: {sub_q}\n\n{result.strip()}")
#                         else:
#                             responses.append(f"### Q{idx+1}: {sub_q}\n\n⚠ No relevant data found.")
#                     except Exception as sub_e:
#                         responses.append(f"### Q{idx+1}: {sub_q}\n\n⚠ Error: {sub_e}")

#                 final_result = "\n\n---\n\n".join(responses)
#             except Exception as e:
#                 final_result = f"⚠ Critical Error: {e}"

#             st.session_state.msgs.append({"role": "assistant", "content": final_result})

# # --- Display Chat ---
# for msg in st.session_state.msgs:
#     with st.chat_message(msg["role"]):
#         st.markdown(msg["content"], unsafe_allow_html=True)














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
import re

# --- Config ---
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_LLM_MODEL = "llama3:latest"
HF_EMBED_MODEL = "intfloat/e5-large"
DB_DIR = "./faiss_db"
TOP_K = 5
ROW_CHUNK_SIZE = 10
TEXT_CHUNK_SIZE = 500

# --- Embeddings ---
embedder = HuggingFaceEmbeddings(model_name=HF_EMBED_MODEL)

# --- Streamlit UI ---
st.set_page_config(page_title="PDF QA", layout="wide")
st.title("Upload your PDF Documents or Scanned Documents in .pdf format")

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

# --- OCR Table and Text Extraction ---
def extract_tables_and_text_from_pdf(pdf_path):
    all_chunks = []
    os.makedirs("tables", exist_ok=True)

    filename = os.path.basename(pdf_path)
    images = convert_from_path(pdf_path, dpi=300)

    for page_num, img in enumerate(images):
        # --- TEXT CHUNKS ---
        full_text = pytesseract.image_to_string(img)
        for i in range(0, len(full_text), TEXT_CHUNK_SIZE):
            text_chunk = full_text[i:i+TEXT_CHUNK_SIZE].strip()
            if text_chunk:
                all_chunks.append(Document(
                    page_content=text_chunk,
                    metadata={"source": filename, "page": page_num + 1}
                ))

        # --- TABLE CHUNKS ---
        ocr_df = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
        if "text" not in ocr_df.columns:
            continue

        ocr_df["text"] = ocr_df["text"].astype(str)
        ocr_df = ocr_df[ocr_df["text"].str.strip() != ""]

        lines = []
        for _, row in ocr_df.iterrows():
            try:
                lines.append((int(row["top"]), row["text"]))
            except:
                continue

        grouped = {}
        for top, word in lines:
            found = False
            for k in grouped.keys():
                if abs(k - top) < 10:
                    grouped[k].append(word)
                    found = True
                    break
            if not found:
                grouped[top] = [word]

        table_rows = list(grouped.values())
        if len(table_rows) < 2:
            continue

        headers = table_rows[0]
        rows = table_rows[1:]
        if not headers or not rows:
            continue

        clean_rows = [r for r in rows if len(r) == len(headers)]
        if not clean_rows:
            continue

        for i in range(0, len(clean_rows), ROW_CHUNK_SIZE):
            partial_rows = clean_rows[i:i+ROW_CHUNK_SIZE]
            formatted_rows = "\n".join([", ".join(r) for r in partial_rows])
            chunk_text = f"Table Chunk\nHeaders: {', '.join(headers)}\nRows:\n{formatted_rows}"

            all_chunks.append(Document(
                page_content=chunk_text,
                metadata={"source": filename, "page": page_num + 1}
            ))

        # Save table JSON
        table_json = {"headers": headers, "rows": clean_rows}
        table_id = str(uuid.uuid4())
        table_path = f"tables/table_{table_id}.json"
        with open(table_path, "w", encoding="utf-8") as f:
            json.dump(table_json, f, indent=2)

    return all_chunks

# --- Index PDFs ---
def process_and_index(files):
    docs = []
    os.makedirs("temp", exist_ok=True)
    for file in files:
        temp_path = os.path.join("temp", file.name)
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        docs.extend(extract_tables_and_text_from_pdf(temp_path))

    if not docs:
        return None

    if Path(DB_DIR).exists():
        vs = FAISS.load_local(DB_DIR, embedder, allow_dangerous_deserialization=True)
        vs.add_documents(docs)
    else:
        vs = FAISS.from_documents(docs, embedder)

    vs.save_local(DB_DIR)
    shutil.rmtree("temp", ignore_errors=True)
    return vs

# --- Format Retrieved Context ---
def format_docs_with_metadata(docs):
    formatted = []
    for doc in docs:
        source = doc.metadata.get("source", "Unknown document")
        page = doc.metadata.get("page", "N/A")
        formatted.append(f"[{source} | Page {page}]\n{doc.page_content}")
    return "\n\n".join(formatted)

# --- LLM Chain ---
def get_chain(vs):
    retriever = vs.as_retriever(search_kwargs={"k": TOP_K})
    context_chain = retriever | format_docs_with_metadata

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful assistant answering questions from documents.

        - Always include the document name and page number (if available) in your response.
        - Respond clearly and use bullet points or tables when appropriate.
        - If no answer is found, say so confidently.

        Context:
        {context}

        Question: {question}
        Answer:
        """
    )

    llm = ChatOllama(model=OLLAMA_LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=0.1)
    return {"context": context_chain, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser()

# --- Sidebar UI ---
with st.sidebar:
    st.image("img/ACL_Digital.png", width=180)
    st.image("img/Cipla_Foundation.png", width=180)
    st.markdown("---")
    st.header("Upload Scanned PDFs")
    uploader = st.file_uploader("Upload scanned PDF(s)", type="pdf", accept_multiple_files=True)
    if uploader:
        st.session_state.uploaded_files = uploader
        for f in uploader:
            st.markdown(f"\u2705 {f.name}")

    if st.button("Extract & Index"):
        if st.session_state.uploaded_files:
            with st.spinner("Extracting tables and text..."):
                vs = process_and_index(st.session_state.uploaded_files)
                if vs:
                    st.session_state.vs = vs
                    st.session_state.msgs = []
                    st.success("Indexing complete!")
                    st.session_state.uploaded_files = []
        else:
            st.warning("Please upload PDFs first.")

    if st.button("Clear Chat"):
        st.session_state.msgs = []

    if st.button("Clear FAISS Index"):
        shutil.rmtree(DB_DIR, ignore_errors=True)
        st.session_state.vs = None
        st.success("FAISS index deleted.")

# --- Load existing index if available ---
if st.session_state.vs is None:
    st.session_state.vs = load_existing_index()

# --- Smart Split ---
def smart_split(query):
    return [q.strip() for q in re.split(r'\b(?:and|or|then|next|also|after that|followed by|&|\n)\b', query, flags=re.IGNORECASE) if q.strip()]

# --- Chat UI ---
st.markdown("### Ask your question")
query = st.chat_input("Ask questions (e.g. 'target and number trained')")

if query:
    if not st.session_state.vs:
        st.error("Please upload and index PDFs first.")
    else:
        sub_queries = smart_split(query)
        st.session_state.msgs.append({"role": "user", "content": query})

        with st.spinner("Thinking through your multi-part query..."):
            try:
                chain = get_chain(st.session_state.vs)
                responses = []

                for idx, sub_q in enumerate(sub_queries):
                    try:
                        result = chain.invoke(sub_q)
                        if result and result.strip():
                            responses.append(f"### Q{idx+1}: {sub_q}\n\n{result.strip()}")
                        else:
                            responses.append(f"### Q{idx+1}: {sub_q}\n\n⚠ No relevant data found.")
                    except Exception as sub_e:
                        responses.append(f"### Q{idx+1}: {sub_q}\n\n⚠ Error: {sub_e}")

                final_result = "\n\n---\n\n".join(responses)
            except Exception as e:
                final_result = f"⚠ Critical Error: {e}"

            st.session_state.msgs.append({"role": "assistant", "content": final_result})

# --- Display Chat ---
for msg in st.session_state.msgs:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)