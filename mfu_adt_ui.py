
# ================================================================
#   MFU ADT Program Guide - Gradio Chat UI
#   Connects to ADT program RAG helpers (mfu_adt_rag.py)
# ================================================================

# pip install gradio langchain-community langchain-text-splitters pypdf faiss-cpu

import os
import re
import time
import requests
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import gradio as gr
from dotenv import load_dotenv

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Backward compatibility if langchain-huggingface is not installed yet.
    from langchain_community.embeddings import HuggingFaceEmbeddings

load_dotenv()

THAILLM_API_KEY = os.getenv("THAILLM_API_KEY", "").strip()
THAILLM_BASE_URL = os.getenv("THAILLM_BASE_URL", "").strip()
THAILLM_MODEL = os.getenv("THAILLM_MODEL", "").strip()
THAILLM_API_HEADER_NAME = os.getenv("THAILLM_API_HEADER_NAME", "apikey").strip()
THAILLM_PAYLOAD_MODEL = os.getenv("THAILLM_PAYLOAD_MODEL", "/model").strip()
PROGRAM_DIRS = [
    Path("Bachelor Degree"),
    Path("Bechelor Degree"),
    Path("Master Degree"),
    Path("PhD Degree"),
]

missing_env = [
    key
    for key, value in [
        ("THAILLM_API_KEY", THAILLM_API_KEY),
        ("THAILLM_BASE_URL", THAILLM_BASE_URL),
        ("THAILLM_MODEL", THAILLM_MODEL),
    ]
    if not value
]
if missing_env:
    raise RuntimeError(
        "Missing required .env value(s): "
        + ", ".join(missing_env)
        + ". Please set them before running mfu_adt_ui.py"
    )


def _build_thaillm_chat_url(base_url: str, model_name: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/{model_name}/v1/chat/completions"


THAILLM_CHAT_URL = _build_thaillm_chat_url(THAILLM_BASE_URL, THAILLM_MODEL)


def load_program_documents():
    documents = []
    pdf_paths = [
        pdf_path
        for program_dir in PROGRAM_DIRS
        for pdf_path in sorted(program_dir.glob("*.pdf"))
    ]

    if not pdf_paths:
        raise FileNotFoundError(
            "No ADT program PDFs found. Expected PDF files in "
            "Bachelor Degree, Bechelor Degree, Master Degree, or PhD Degree."
        )

    for pdf_path in pdf_paths:
        loaded_docs = PyPDFLoader(str(pdf_path)).load()
        for doc in loaded_docs:
            doc.metadata["title"] = pdf_path.stem
            doc.metadata["degree_level"] = pdf_path.parent.name
        documents.extend(loaded_docs)

    return documents


# ----------------------------------------------------------------
# 1. BUILD RAG PIPELINE (runs once at startup)
# ----------------------------------------------------------------
print("Initializing MFU ADT Program RAG pipeline...")

# -- Load & split document
documents = load_program_documents()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"  -> {len(chunks)} chunks created")

# -- Embeddings + FAISS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
print("  -> Vector store ready")

# -- Prompt
SYSTEM_PROMPT = """You are a friendly MFU ADT Program Guide assistant for Mae Fah Luang University.
Your job is to help students and applicants understand programs from the School of Applied Digital Technology based only on the provided context.
Answer clearly and concisely. Mention the relevant program name and degree level when possible.
If the answer is not in the context, say "I don't have that information right now."
"""

print(f"  -> ThaiLLM endpoint: {THAILLM_CHAT_URL}")
print("  -> RAG pipeline ready\n")

# ----------------------------------------------------------------
# 2. CHAT FUNCTION
# ----------------------------------------------------------------

def _call_thaillm(question: str, source_docs):
    context_str = "\n\n".join(
        f"--- เอกสาร {i + 1} ({doc.metadata.get('title') or doc.metadata.get('source') or f'Chunk {i + 1}'}) ---\n{doc.page_content}"
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
            resp = requests.post(
                THAILLM_CHAT_URL,
                headers=headers,
                json=payload,
                timeout=60,
            )
            if resp.status_code == 429:
                time.sleep(min(2**attempt, 15))
                continue
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            return re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL).strip()
        except requests.HTTPError as err:
            last_error = f"HTTP {err.response.status_code}: {err.response.text[:300]}"
            if err.response is not None and err.response.status_code in (401, 403):
                break
        except Exception as err:
            last_error = str(err)
        time.sleep(2**attempt)

    raise RuntimeError(last_error or "Unknown ThaiLLM error")


def chat(message, history):
    if not message.strip():
        return "Please type a question about ADT programs at MFU."

    try:
        source_docs = retriever.invoke(message)
        answer = _call_thaillm(message, source_docs)
    except Exception as err:
        return (
            "Request blocked or failed at ThaiLLM endpoint.\n\n"
            f"Details: {err}\n\n"
            "Please verify `THAILLM_BASE_URL`, `THAILLM_MODEL`, "
            "`THAILLM_PAYLOAD_MODEL`, `THAILLM_API_KEY`, and "
            "`THAILLM_API_HEADER_NAME` in `.env`."
        )

    # Append source snippets
    if source_docs:
        source_text = "\n\n---\n**Sources used:**\n"
        for i, doc in enumerate(source_docs, 1):
            snippet = doc.page_content[:100].strip().replace("\n", " ")
            title = doc.metadata.get("title") or doc.metadata.get("source") or f"Source {i}"
            source_text += f"- [{i}] {title}: {snippet}...\n"
        answer += source_text

    return answer


# ----------------------------------------------------------------
# 3. GRADIO CHAT INTERFACE
# ----------------------------------------------------------------
with gr.Blocks(title="BDA_Project2_Group8") as demo:

    gr.Markdown(
        """
        # BDA_Project2_Group8
        ## MFU ADT Program Guide — RAG Chatbot
        ### School of Applied Digital Technology · Mae Fah Luang University

        **Group 8 Members:**
        | Student ID | Student Name |
        |---|---|
        | 6631501152 | Nattapat Ismael |
        | 6631501124 | Athinan Singkaew |
        | 6631501185 | JAYVERT DALE ANCHETA |
        | 6631501132 | Htet Lin Aung |
        | 6631501139 | Sai Myat Thura Koe |

        Ask about degree programs, admission requirements, curriculum, study plans, and career paths.
        """
    )

    gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(
            height=480,
            placeholder="<b>Hi! I'm your MFU ADT Program Guide.</b><br>Try asking: <i>'What bachelor programs are available?'</i>",
        ),
        textbox=gr.Textbox(
            placeholder="Ask about ADT programs at MFU...",
            container=False,
            scale=7
        ),
        examples=[
            "What bachelor programs are available in the School of Applied Digital Technology?",
            "Compare Software Engineering and Computer Engineering.",
            "What careers can students pursue after Digital Technology for Business Innovation?",
            "What master degree programs are available?",
            "Tell me about the PhD in Computer Engineering.",
        ],
    )

    gr.Markdown(
        """
        ---
        *Powered by ThaiLLM · LangChain · FAISS · all-MiniLM-L6-v2*
        """
    )

# ----------------------------------------------------------------
# 4. LAUNCH
# ----------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(share=True, theme=gr.themes.Soft())
