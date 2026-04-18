
# ================================================================
#   MFU Cafe Guide - RAG System
#   Model  : openthaigpt/openthaigpt1.0-7b-chat
#   Embed  : sentence-transformers/all-MiniLM-L6-v2
#   Vector : FAISS
# ================================================================

# pip install langchain langchain-community faiss-cpu
#             sentence-transformers transformers accelerate

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

# ----------------------------------------------------------------
# 1. LOAD DOCUMENT
# ----------------------------------------------------------------
print("Loading document...")
loader = TextLoader("mfu_cafes.txt", encoding="utf-8")
documents = loader.load()

# ----------------------------------------------------------------
# 2. SPLIT TEXT
# ----------------------------------------------------------------
print("Splitting text...")
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
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
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
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
# 5. SYSTEM PROMPT — MFU Cafe Guide
# ----------------------------------------------------------------
SYSTEM_PROMPT = """You are a friendly MFU Cafe Guide assistant for Mae Fah Luang University.
Your job is to help students and visitors find the best cafes near MFU based on the context provided.
Answer in a helpful, concise, and friendly tone.
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
print("   MFU CAFE GUIDE - Ask me anything about cafes near MFU!")
print("   Type 'exit' to quit.")
print("=" * 60)

while True:
    question = input("\nYou: ").strip()
    if question.lower() in ["exit", "quit", "q"]:
        print("Goodbye! Enjoy your coffee! :)")
        break
    if not question:
        continue

    result = qa_chain.invoke({"query": question})
    print(f"\nGuide: {result['result']}")

    # Optional: show source chunks used
    print("\n[Sources used]")
    for i, doc in enumerate(result["source_documents"], 1):
        print(f"  [{i}] {doc.page_content[:100].strip()}...")
