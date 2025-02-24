import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings  # Alternative embeddings
from langchain_community.llms import Together
from langchain.chains import RetrievalQA

# Set Together API key
os.environ["TOGETHER_API_KEY"] = "e5d09114c4d8bace3b652f59e6a6c36d5235673a153730dbedee257d21b8902b"  # Replace with your actual API key

# Load text file as knowledge base
def load_text_data(file_path):
    loader = TextLoader(file_path)
    documents = loader.load()
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_documents(documents)
    
    # Use Hugging Face embeddings (since TogetherEmbeddings doesn't exist)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(docs, embeddings)
    
    return vector_store

# Initialize chatbot with RAG
def initialize_chatbot(vector_store):
    retriever = vector_store.as_retriever()
    llm = Together(model="togethercomputer/llama-3-8b")  # Use Together LLM
    return RetrievalQA.from_chain_type(llm, retriever=retriever)

# Streamlit UI
st.title("🤖 RAG Chatbot with Together AI")
st.write("This chatbot retrieves relevant info from a text file and generates responses using Together AI.")

# Upload text file
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file:
    file_path = "C:/Users/Acer/Desktop/space.txt"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Load and process text file
    vector_store = load_text_data(file_path)
    qa_chain = initialize_chatbot(vector_store)
    
    # Chat interface
    st.subheader("Chat with AI")
    user_input = st.text_input("Ask something:")
    
    if st.button("Send") and user_input:
        response = qa_chain.run(user_input)
        st.write("🤖 AI:", response)
