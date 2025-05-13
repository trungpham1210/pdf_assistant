import streamlit as st
import fitz  # PyMuPDF
import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

load_dotenv()

st.set_page_config(page_title="PDF Q&A Assistant", layout="wide")
st.title("🤖 Ask Questions About Your PDF (Hugging Face Edition)")

pdf = st.file_uploader("Upload a PDF file", type="pdf")

if pdf:
    # Extract text from PDF
    doc = fitz.open(stream=pdf.read(), filetype="pdf")
    text = "".join([page.get_text() for page in doc])

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    chunks = splitter.split_text(text)

    # Create embeddings using simple FAISS (no LLM needed)
    from langchain.embeddings import HuggingFaceEmbeddings
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    # Set up Hugging Face pipeline
    hf_pipeline = pipeline("text2text-generation", model="google/flan-t5-base", max_length=256)
    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    # Create the QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        return_source_documents=False
    )

    # Ask questions
    question = st.text_input("Ask a question about this document:")
    if question:
        answer = qa_chain.run(question)
        st.markdown(f"**Answer:** {answer}")
