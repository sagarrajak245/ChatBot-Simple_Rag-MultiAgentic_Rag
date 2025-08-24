import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_community.vectorstores import FAISS  
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

# ================================
# 1️⃣ Load environment variables
# ================================
# Ensure your .env file contains:
# GROQ_API_KEY="your_groq_api_key_here"
load_dotenv()

# ================================
# 2️⃣ Streamlit UI setup
# ================================
st.set_page_config(page_title="📄 RAG Chatbot with Groq & FAISS")
st.title("📄 RAG Chatbot with Groq & FAISS")
st.write("Upload a PDF and ask questions about it using Groq's LLaMA 3.3 model.")

# ================================
# 3️⃣ File upload
# ================================
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file is not None:
    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        temp_pdf_path = tmp_file.name

    # Load and process PDF
    loader = PyPDFLoader(temp_pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(docs)

    # ================================
    # 4️⃣ Create embeddings
    # ================================
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ================================
    # 5️⃣ Create FAISS vector store
    # ================================
    vectorstore = FAISS.from_documents(chunks, embeddings)

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # ================================
    # 6️⃣ Define custom prompt
    # ================================
    prompt_template = """
    You are a helpful assistant. Use ONLY the provided context to answer the question.
    If the answer is not in the context, say: "I don't know."

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    # ================================
    # 7️⃣ Initialize Groq LLM
    # ================================
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )

    # ================================
    # 8️⃣ Create RetrievalQA chain
    # ================================
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": prompt}
    )

    # ================================
    # 9️⃣ Question Input
    # ================================
    st.subheader("Ask a Question about Your PDF")
    user_question = st.text_input("Enter your question:")

    if user_question:
        with st.spinner("Generating answer..."):
            answer = qa_chain.run(user_question)
        st.write("**Answer:**", answer)
