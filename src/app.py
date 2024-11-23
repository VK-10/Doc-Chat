import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import CharacterTextSplitter  # Corrected import
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain  # Corrected import
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

def load_document_loader(file_path):
    loader = PyPDFLoader(file_path=file_path)
    documents = loader.load()
    return documents

def setup_vector_store(documents):
    embedding = HuggingFaceEmbeddings()
    text_splitter = CharacterTextSplitter(
        separator="\n",  # Corrected spelling
        chunk_size=1000,
        chunk_overlap=200
    )

    doc_chunks = text_splitter.split_documents(documents)
    vector_store = FAISS.from_documents(doc_chunks, embedding)
    return vector_store

def create_chain(vector_store):
    llm = ChatGroq(
        model_name="llama-3.2-3b-preview",  # Updated model name
        temperature=0.2,
    )

    retriever = vector_store.as_retriever()
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    chain = ConversationalRetrievalChain.from_llm(  # Corrected chain creation
        llm=llm,
        retriever=retriever,
        memory=memory,
        verbose=True
    )

    return chain

st.set_page_config(
    page_title="Doc-chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Doc-chat with Llama 2")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    file_path = os.path.join(working_dir, uploaded_file.name)  # Safer path joining
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    if "vectorstore" not in st.session_state:
        documents = load_document_loader(file_path)
        st.session_state.vectorstore = setup_vector_store(documents)

    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = create_chain(st.session_state.vectorstore)

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["message"])  # Corrected dictionary access

user_input = st.chat_input("Ask something...")  # Changed to chat_input for better UI

if user_input:
    st.session_state.chat_history.append({"role": "user", "message": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = st.session_state.conversation_chain({"question": user_input})
        assistant_message = response["answer"]
        st.markdown(assistant_message)
        st.session_state.chat_history.append({"role": "assistant", "message": assistant_message})