import os
import streamlit as st
import tempfile
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama.chat_models import ChatOllama
from langchain_community.vectorstores import Chroma
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import WebBaseLoader, CSVLoader, JSONLoader, PyPDFLoader


st.title("📚 RAG Chatbot with Dynamic Inputs")

# 1. URLs
url_input = st.text_area("Enter URLs (one per line):")
urls = [u.strip() for u in url_input.split("\n") if u.strip()]

# 2. File Upload
uploaded_files = st.file_uploader("Upload CSV or PDF files", type=["csv", "pdf"], accept_multiple_files=True)

if st.button("Ingest & Build VectorDB"):
    all_docs = []

    # Load from URLs
    if urls:
        st.info("🔗 Loading from URLs...")
        web_loader = WebBaseLoader(urls)
        all_docs.extend(web_loader.load())

    # Load from uploaded files
    for file in uploaded_files:
        file_type = file.type
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        if file_type == "text/csv":
            loader = CSVLoader(file_path=tmp_path)
        elif file_type == "application/pdf":
            loader = PyPDFLoader(file_path=tmp_path)
        else:
            st.warning(f"Unsupported file type: {file_type}")
            continue

        all_docs.extend(loader.load())

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    splits = text_splitter.split_documents(all_docs)

    # Vectorstore
    persist_directory = "./chroma_db"
    embeddings = OllamaEmbeddings(model='mxbai-embed-large')
    vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=persist_directory)
    vectorstore.persist()

    st.success("✅ Data ingested and stored!")

    # Save state
    st.session_state["vectorstore"] = vectorstore

# 3. Chat Section
if "vectorstore" in st.session_state:
    query = st.text_input("Ask a question about the uploaded content:")
    if query:
        retriever = st.session_state["vectorstore"].as_retriever()
        llm = ChatOllama(model="llama3")
        from langchain.chains import RetrievalQA

        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        answer = qa_chain.run(query)

        st.markdown("### 💬 Answer:")
        st.write(answer)