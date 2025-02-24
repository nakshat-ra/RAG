import os
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Set Together API Key
os.environ["TOGETHER_API_KEY"] = "e5d09114c4d8bace3b652f59e6a6c36d5235673a153730dbedee257d21b8902b"  # Replace with actual Together API Key

# Streamlit UI
st.title("🤖 RAG Chatbot with Together API")
st.write("Upload a text file for the AI to use as knowledge.")

# Upload text file
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    temp_file_path = "C:/Users/Acer/Desktop/space.txt"
    with open(temp_file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("File uploaded successfully!")

    # Load and process the text file
    def load_text_data(file_path):
        loader = TextLoader(file_path)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        docs = text_splitter.split_documents(documents)

        # Use Hugging Face Embeddings (FREE)
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISS.from_documents(docs, embeddings)
        return vector_store

    vector_store = load_text_data(temp_file_path)
    retriever = vector_store.as_retriever()

    # Use Together API for LLM
    llm = ChatOpenAI(model_name="togethercomputer/llama-2-7b-chat")

    # Create RAG Chain
    qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

    # Chat interface
    user_input = st.text_input("Ask something:")
    if st.button("Send") and user_input:
        response = qa_chain.run(user_input)
        st.write("🤖 AI:", response)
