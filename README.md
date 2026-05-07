---
title: BDA Project2 Group8
emoji: 🎓
colorFrom: blue
colorTo: green
sdk: streamlit
sdk_version: 1.45.0
app_file: streamlit_app.py
pinned: false
---

# MFU ADT Program Guide — RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about programs from the **School of Applied Digital Technology (ADT)** at Mae Fah Luang University.

The system reads PDF program brochures for Bachelor, Master, and PhD degrees, builds a FAISS vector store, and answers questions using the **ThaiLLM API** (Gradio UI) or a local **OpenThaiGPT** model (CLI).

---

## Tech Stack

| Layer | Tool |
|---|---|
| PDF Loading | LangChain `PyPDFLoader` |
| Text Splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | FAISS |
| LLM (UI) | ThaiLLM API (OpenAI-compatible) |
| LLM (CLI) | `openthaigpt/openthaigpt1.0-7b-chat` (local) |
| UI | Gradio |

---

## Project Structure

```
├── Bechelor Degree/        # Bachelor degree PDF brochures
├── Master Degree/          # Master degree PDF brochures
├── PhD Degree/             # PhD degree PDF brochures
├── mfu_adt_ui.py           # Gradio web UI (uses ThaiLLM API)
├── mfu_adt_rag.py          # CLI version (uses local OpenThaiGPT)
├── requirements.txt
├── .env                    # API keys (not committed)
└── .env.example            # Template for .env
```

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/ntpism17/ADT_RAG.git
cd ADT_RAG
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env` and fill in your ThaiLLM API key:

```
THAILLM_API_KEY=your_api_key_here
THAILLM_BASE_URL=https://thaillm.or.th/api/kbtg/v1
THAILLM_MODEL=kbtg
THAILLM_API_HEADER_NAME=apikey
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

Verify:

```bash
python -c "import requests, gradio, langchain_community; print('OK')"
```

---

## Usage

### Gradio Web UI (recommended)

Uses the ThaiLLM API — no GPU required.

```bash
python mfu_adt_ui.py
```

Open the local URL shown in the terminal (or the public share link).

### CLI (local model)

Uses OpenThaiGPT locally — requires a GPU or sufficient RAM.

```bash
python mfu_adt_rag.py
```

Type your question and press Enter. Type `exit` to quit.

---

## Available Programs

**Bachelor Degree**
- Computer Engineering
- Digital and Communication Engineering
- Digital Technology for Business Innovation
- Multimedia Technology and Animation
- Software Engineering

**Master Degree**
- Computer Engineering
- Digital Transformation Technology

**PhD Degree**
- Computer Engineering

---

*Powered by ThaiLLM · LangChain · FAISS · all-MiniLM-L6-v2*
