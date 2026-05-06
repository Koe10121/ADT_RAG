
# ================================================================
#   MFU ADT Program Guide - RAG System
#   Model  : openthaigpt/openthaigpt1.0-7b-chat
#   Embed  : sentence-transformers/all-MiniLM-L6-v2
#   Vector : FAISS
# ================================================================

# pip install langchain langchain-community pypdf faiss-cpu
#             sentence-transformers transformers accelerate

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    # Backward compatibility if langchain-huggingface is not installed yet.
    from langchain_community.embeddings import HuggingFaceEmbeddings

PROGRAM_DIRS = [
    Path("Bachelor Degree"),
    Path("Bechelor Degree"),
    Path("Master Degree"),
    Path("PhD Degree"),
]


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
        print(f"  -> Loading {pdf_path}")
        loaded_docs = PyPDFLoader(str(pdf_path)).load()
        for doc in loaded_docs:
            doc.metadata["title"] = pdf_path.stem
            doc.metadata["degree_level"] = pdf_path.parent.name
        documents.extend(loaded_docs)

    return documents

# ----------------------------------------------------------------
# 1. LOAD DOCUMENT
# ----------------------------------------------------------------
print("Loading ADT program documents...")
documents = load_program_documents()

# ----------------------------------------------------------------
# 2. SPLIT TEXT
# ----------------------------------------------------------------
print("Splitting text...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=120,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"  -> {len(chunks)} chunks created")

# ----------------------------------------------------------------
# 3. EMBEDDINGS + VECTOR STORE (FAISS)
# ----------------------------------------------------------------
print("Creating embeddings and FAISS vector store...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
print("  -> Vector store ready")

# ----------------------------------------------------------------
# 4. LOAD OPENTHAIGPT MODEL
# ----------------------------------------------------------------
print("Loading OpenThaiGPT model (this may take a while)...")
MODEL_ID = "openthaigpt/openthaigpt1.0-7b-chat"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    repetition_penalty=1.1,
    do_sample=True,
)

llm = HuggingFacePipeline(pipeline=pipe)
print("  -> Model ready")

# ----------------------------------------------------------------
# 5. SYSTEM PROMPT - MFU ADT Program Guide
# ----------------------------------------------------------------
SYSTEM_PROMPT = """You are a friendly MFU ADT Program Guide assistant for Mae Fah Luang University.
Your job is to help students and applicants understand programs from the School of Applied Digital Technology based only on the context provided.
Answer clearly and concisely. Mention the relevant program name and degree level when possible.
If the answer is not in the context, say "I don't have that information right now."

Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=SYSTEM_PROMPT,
    input_variables=["context", "question"]
)

# ----------------------------------------------------------------
# 6. RETRIEVAL QA CHAIN
# ----------------------------------------------------------------
print("Building retrieval chain...")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
print("  -> Chain ready\n")

# ----------------------------------------------------------------
# 7. INTERACTIVE Q&A LOOP
# ----------------------------------------------------------------
print("=" * 60)
print("   MFU ADT PROGRAM GUIDE - Ask me anything about ADT programs!")
print("   Type 'exit' to quit.")
print("=" * 60)

while True:
    question = input("\nYou: ").strip()
    if question.lower() in ["exit", "quit", "q"]:
        print("Goodbye! Good luck with your studies. :)")
        break
    if not question:
        continue

    result = qa_chain.invoke({"query": question})
    print(f"\nADT Guide: {result['result']}")

    # Optional: show source chunks used
    print("\n[Sources used]")
    for i, doc in enumerate(result["source_documents"], 1):
        title = doc.metadata.get("title") or doc.metadata.get("source") or "Unknown source"
        print(f"  [{i}] {title}: {doc.page_content[:100].strip()}...")
