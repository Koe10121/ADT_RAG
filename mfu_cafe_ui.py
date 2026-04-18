
# ================================================================
#   MFU Cafe Recommender - Gradio Chat UI
#   Connects to LangChain RAG pipeline (mfu_cafe_rag.py)
# ================================================================

# pip install gradio

import gradio as gr
import torch

from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ----------------------------------------------------------------
# 1. BUILD RAG PIPELINE (runs once at startup)
# ----------------------------------------------------------------
print("Initializing MFU Cafe RAG pipeline...")

# -- Load & split document
loader = TextLoader("mfu_cafes.txt", encoding="utf-8")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
    separators=["\n\n", "\n", " ", ""]
)
chunks = splitter.split_documents(documents)
print(f"  -> {len(chunks)} chunks created")

# -- Embeddings + FAISS
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"}
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
print("  -> Vector store ready")

# -- Load OpenThaiGPT
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

# -- Prompt
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

# -- QA Chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
print("  -> RAG chain ready\n")

# ----------------------------------------------------------------
# 2. CHAT FUNCTION
# ----------------------------------------------------------------


def chat(message, history):
    if not message.strip():
        return "Please type a question about cafes near MFU."

    result = qa_chain.invoke({"query": message})
    answer = result["result"].strip()

    # Append source snippets
    sources = result.get("source_documents", [])
    if sources:
        source_text = "\n\n---\n**Sources used:**\n"
        for i, doc in enumerate(sources, 1):
            snippet = doc.page_content[:100].strip().replace("\n", " ")
            source_text += f"- [{i}] {snippet}...\n"
        answer += source_text

    return answer


# ----------------------------------------------------------------
# 3. GRADIO CHAT INTERFACE
# ----------------------------------------------------------------
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="MFU Cafe Recommender"
) as demo:

    gr.Markdown(
        """
        # MFU Cafe Recommender
        ### Your AI-powered cafe guide near Mae Fah Luang University
        Ask me anything — best cafe for studying, photo spots, opening hours, recommended menus, and more!
        """
    )

    chatbot = gr.ChatInterface(
        fn=chat,
        chatbot=gr.Chatbot(
            height=480,
            placeholder="<b>Hi! I'm your MFU Cafe Guide.</b><br>Try asking: <i>'Which cafe is best for studying?'</i>",
        ),
        textbox=gr.Textbox(
            placeholder="Ask about cafes near MFU...",
            container=False,
            scale=7
        ),
        examples=[
            "Which cafe is best for studying?",
            "Where can I take good photos near MFU?",
            "What are the opening hours of The Cloud Cafe?",
            "What is the recommended menu at Doi Tung Hillside Coffee?",
            "Which cafe is closest to the MFU main gate?",
        ],
        retry_btn="Retry",
        undo_btn="Undo",
        clear_btn="Clear Chat",
    )

    gr.Markdown(
        """
        ---
        *Powered by OpenThaiGPT · LangChain · FAISS · all-MiniLM-L6-v2*
        """
    )

# ----------------------------------------------------------------
# 4. LAUNCH
# ----------------------------------------------------------------
if __name__ == "__main__":
    demo.launch(share=True)
