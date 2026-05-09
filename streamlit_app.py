# ================================================================
#   BDA Project 2 — Group 8
#   MFU ADT Program Guide — RAG Chatbot (Streamlit UI)
#
#   Group Members:
#   6631501152  Nattapat Ismael
#   6631501124  Athinan Singkaew
#   6631501185  JAYVERT DALE ANCHETA
#   6631501132  Htet Lin Aung
#   6631501139  Sai Myat Thura Koe
# ================================================================

import os
import re
import time
import requests
import streamlit as st
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
load_dotenv()

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings


def _get_secret(key: str, default: str = "") -> str:
    """Read from Streamlit Secrets first, fall back to env var."""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return os.environ.get(key, default)


THAILLM_API_KEY         = _get_secret("THAILLM_API_KEY").strip()
THAILLM_BASE_URL        = _get_secret("THAILLM_BASE_URL").strip()
THAILLM_MODEL           = _get_secret("THAILLM_MODEL").strip()
THAILLM_API_HEADER_NAME = _get_secret("THAILLM_API_HEADER_NAME", "apikey").strip()
THAILLM_PAYLOAD_MODEL   = _get_secret("THAILLM_PAYLOAD_MODEL", "/model").strip()

PROGRAM_DIRS = [
    Path("Bachelor Degree"),
    Path("Bechelor Degree"),
    Path("Master Degree"),
    Path("PhD Degree"),
]

SYSTEM_PROMPT = """You are a friendly MFU ADT Program Guide assistant for Mae Fah Luang University.
Your job is to help students and applicants understand programs from the School of Applied Digital Technology based only on the provided context.
Answer clearly and concisely. Mention the relevant program name and degree level when possible.
If the answer is not in the context, say "I don't have that information right now."
"""


def _build_thaillm_chat_url(base_url: str, model_name: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/{model_name}/v1/chat/completions"


@st.cache_resource(show_spinner="Loading ADT program documents and building index...")
def build_rag_pipeline():
    documents = []
    pdf_paths = [
        pdf_path
        for program_dir in PROGRAM_DIRS
        for pdf_path in sorted(program_dir.glob("*.pdf"))
    ]
    if not pdf_paths:
        st.error("No PDF files found in degree folders.")
        st.stop()

    for pdf_path in pdf_paths:
        loaded_docs = PyPDFLoader(str(pdf_path)).load()
        for doc in loaded_docs:
            doc.metadata["title"] = pdf_path.stem
            doc.metadata["degree_level"] = pdf_path.parent.name
        documents.extend(loaded_docs)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": 5})


def call_thaillm(question: str, source_docs) -> str:
    chat_url = _build_thaillm_chat_url(THAILLM_BASE_URL, THAILLM_MODEL)
    context_str = "\n\n".join(
        f"--- เอกสาร {i + 1} ({doc.metadata.get('title') or f'Chunk {i+1}'}) ---\n{doc.page_content}"
        for i, doc in enumerate(source_docs)
    )
    user_prompt = f"Context:\n{context_str}\n\n---\nคำถาม: {question}"

    headers = {
        "Content-Type": "application/json",
        THAILLM_API_HEADER_NAME: THAILLM_API_KEY,
    }
    payload = {
        "model": THAILLM_PAYLOAD_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "max_tokens": 512,
        "temperature": 0.3,
    }

    last_error = None
    for attempt in range(4):
        try:
            resp = requests.post(chat_url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                time.sleep(min(2 ** attempt, 15))
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        except requests.HTTPError as err:
            last_error = f"HTTP {err.response.status_code}: {err.response.text[:300]}"
            if err.response.status_code in (401, 403):
                break
        except Exception as err:
            last_error = str(err)
        time.sleep(2 ** attempt)

    raise RuntimeError(last_error or "Unknown ThaiLLM error")


# ----------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------
st.set_page_config(
    page_title="BDA_Project2_Group8",
    page_icon="🎓",
    layout="centered",
)

# ----------------------------------------------------------------
# SIDEBAR — group info
# ----------------------------------------------------------------
with st.sidebar:
    st.title("BDA_Project2_Group8")
    st.markdown("**MFU ADT Program Guide**")
    st.markdown("School of Applied Digital Technology  \nMae Fah Luang University")
    st.divider()
    st.markdown("**Group 8 Members**")
    st.markdown(
        """
| Student ID | Name |
|---|---|
| 6631501152 | Nattapat Ismael |
| 6631501124 | Athinan Singkaew |
| 6631501185 | JAYVERT DALE ANCHETA |
| 6631501132 | Htet Lin Aung |
| 6631501139 | Sai Myat Thura Koe |
        """
    )
    st.divider()
    st.caption("Powered by ThaiLLM · LangChain · FAISS · all-MiniLM-L6-v2")

# ----------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------
st.title("🎓 BDA_Project2_Group8")
st.subheader("MFU ADT Program Guide")
st.caption("Ask anything about programs from the School of Applied Digital Technology.")

# ----------------------------------------------------------------
# EXAMPLE QUESTIONS
# ----------------------------------------------------------------
EXAMPLES = [
    "What bachelor programs are available in ADT?",
    "Compare Software Engineering and Computer Engineering.",
    "What careers after Digital Technology for Business Innovation?",
    "What master degree programs are available?",
    "Tell me about the PhD in Computer Engineering.",
]

with st.expander("💡 Example questions"):
    for ex in EXAMPLES:
        if st.button(ex, key=ex):
            st.session_state.pending_question = ex

# ----------------------------------------------------------------
# BUILD RAG (cached — runs once)
# ----------------------------------------------------------------
retriever = build_rag_pipeline()

# ----------------------------------------------------------------
# CHAT STATE
# ----------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

def generate_answer(prompt: str) -> str:
    try:
        source_docs = retriever.invoke(prompt)
        answer = call_thaillm(prompt, source_docs)
        if source_docs:
            answer += "\n\n---\n**Sources:**\n"
            for i, doc in enumerate(source_docs, 1):
                title = doc.metadata.get("title") or f"Source {i}"
                snippet = doc.page_content[:100].strip().replace("\n", " ")
                answer += f"- [{i}] {title}: {snippet}...\n"
        return answer
    except Exception as err:
        return (
            f"âŒ Request failed: {err}\n\n"
            "Check your API credentials in Streamlit Secrets."
        )

pending_prompt = st.session_state.pop("pending_question", None)
typed_prompt = st.chat_input("Ask about ADT programs at MFU...")
prompt = pending_prompt or typed_prompt

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ----------------------------------------------------------------
# CHAT INPUT
# ----------------------------------------------------------------
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                source_docs = retriever.invoke(prompt)
                answer = call_thaillm(prompt, source_docs)
                if source_docs:
                    answer += "\n\n---\n**Sources:**\n"
                    for i, doc in enumerate(source_docs, 1):
                        title = doc.metadata.get("title") or f"Source {i}"
                        snippet = doc.page_content[:100].strip().replace("\n", " ")
                        answer += f"- [{i}] {title}: {snippet}...\n"
            except Exception as err:
                answer = (
                    f"❌ Request failed: {err}\n\n"
                    "Check your API credentials in Streamlit Secrets."
                )
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
