import streamlit as st
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Together

# GitHub raw URL of the file
GITHUB_FILE_URL = "https://raw.githubusercontent.com/your-username/your-repo/main/space.txt"  # Replace with your actual URL

st.title("🤖 RAG Chatbot with Together API")

try:
    # Fetch the file content from GitHub
    response = requests.get(GITHUB_FILE_URL)
    response.raise_for_status()  # Raise error if request fails
    file_content = response.text  # Read the file content

    st.success("File loaded successfully from GitHub!")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.create_documents([file_content])
    
    # Create FAISS vector store
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    
    # Initialize Together API model
    llm = Together(model="togethercomputer/mistral-7b-instruct")

    # Create RAG chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=vector_store.as_retriever())

    # Chat interface
    user_input = st.text_input("Ask something:")
    if st.button("Send") and user_input:
        response = qa_chain.run(user_input)
        st.write("🤖 AI:", response)

except requests.exceptions.RequestException as e:
    st.error(f"Error loading file: {e}")
