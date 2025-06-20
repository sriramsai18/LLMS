import argparse
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

VECTORSTORE_PATH = "faiss_legal_index"

def load_local_llm(model_id="TinyLlama/TinyLlama-1.1B-chat-v1.0"):
    print("[INFO] Loading local HuggingFace LLM...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_new_tokens=512)
    return HuggingFacePipeline(pipeline=pipe)

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def build_vectorstore(chunks, embeddings):
    vs = FAISS.from_texts(chunks, embeddings)
    vs.save_local(VECTORSTORE_PATH)
    return vs

def load_vectorstore(embeddings):
    return FAISS.load_local(VECTORSTORE_PATH, embeddings)

def process_pdf(pdf_path):
    print(f"[INFO] Loading PDF: {pdf_path}")
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            raw_text += page_text

    if not raw_text.strip():
        raise ValueError("No text found in the PDF. Is it a scanned image or empty?")

    print("[INFO] Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)

    print("[INFO] Loading embeddings...")
    embeddings = load_embeddings()
    print("[INFO] Creating new vectorstore (skipping .pkl file)...")
    vectorstore = FAISS.from_texts(chunks, embeddings)
    return vectorstore

def ask_question(vectorstore, question, llm):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([f"---\n{doc.page_content}" for doc in relevant_docs])

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
You are a legal assistant AI. Use the CONTEXT below to answer the QUESTION.
Think step-by-step, and include citations by quoting relevant parts of the context.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:""" )

    chain: RunnableSequence = prompt | llm

    print("\n[INFO] Getting answer from local LLM...")
    response = chain.invoke({"context": context, "question": question})
    print("\n📘 Answer:")
    print(response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Document Q&A with Local LLM")
    parser.add_argument("pdf_path", help="Path to the PDF legal document")
    args = parser.parse_args()

    try:
        vectorstore = process_pdf(args.pdf_path)
        llm = load_local_llm()

        while True:
            question = input("\n🔍 Enter your question (or type 'exit' to quit):\n> ")
            if question.lower().strip() in ["exit", "quit"]:
                break
            ask_question(vectorstore, question, llm)

    except Exception as e:
        print(f"[ERROR] {e}")