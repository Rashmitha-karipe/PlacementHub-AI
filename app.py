import os
from dotenv import load_dotenv

import streamlit as st

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQA

from langchain_core.documents import Document

# ---------------------------------
# Load Environment Variables
# ---------------------------------
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

# ---------------------------------
# Streamlit Page Config
# ---------------------------------
st.set_page_config(page_title="PlacementHub AI")

st.title("🎯 PlacementHub AI")
st.subheader("Company-wise Interview Preparation Assistant")

# ---------------------------------
# Company Selection
# ---------------------------------
company = st.selectbox(
    "Select Company",
    ["amazon", "tcs", "infosys"]
)

# ---------------------------------
# Load Documents
# ---------------------------------
data_path = f"data/{company}"

documents = []

if os.path.exists(data_path):

    for file in os.listdir(data_path):

        file_path = os.path.join(data_path, file)

        # PDF Files
        if file.endswith(".pdf"):

            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())

        # TXT Files
        elif file.endswith(".txt"):

            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()

            documents.append(
                Document(page_content=text)
            )

# ---------------------------------
# Check Documents
# ---------------------------------
if len(documents) == 0:

    st.error("No documents found inside selected company folder.")

else:

    # ---------------------------------
    # Split Text
    # ---------------------------------
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    docs = text_splitter.split_documents(documents)

    # ---------------------------------
    # Embeddings
    # ---------------------------------
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ---------------------------------
    # Create Vector Store
    # ---------------------------------
    vectorstore = FAISS.from_documents(
        docs,
        embeddings
    )

    retriever = vectorstore.as_retriever()

    # ---------------------------------
    # Load LLM
    # ---------------------------------
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama-3.1-8b-instant"
    )

    # ---------------------------------
    # Create QA Chain
    # ---------------------------------
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever
    )

    # ---------------------------------
    # User Input
    # ---------------------------------
    query = st.text_input("Ask Interview Questions")

    if query:

        response = qa.run(query)

        st.write("### Answer")
        st.write(response)