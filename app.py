import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import tempfile

def init_session_state():
    """Initialize session state variables"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "chain" not in st.session_state:
        st.session_state.chain = None

def create_chain(vectorstore, llm):
    """Create conversation chain with correct output key configuration"""
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key="answer"  # Explicitly set output key
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": None},  # Use default prompt
        verbose=True
    )
    
    return chain

def main():
    st.title("📚 Document Chat Assistant")
    
    # Initialize session state
    init_session_state()
    
    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    
    if not api_key:
        st.error("Please set GROQ_API_KEY in your environment variables")
        st.stop()
        
    # Initialize LLM
    llm = ChatGroq(
        model_name="llama-3.2-3b-preview",
        temperature=0.2,
        api_key=api_key
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"])

    if uploaded_file:
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                file_path = tmp_file.name

            # Load and process document
            loader = PyPDFLoader(file_path)
            documents = loader.load()
            
            # Split text
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(documents)
            
            # Create vector store
            embeddings = HuggingFaceEmbeddings()
            vectorstore = FAISS.from_documents(chunks, embeddings)
            
            # Create conversation chain
            st.session_state.chain = create_chain(vectorstore, llm)
            
            # Cleanup temporary file
            os.unlink(file_path)
            
            st.success("Document processed successfully! You can now ask questions.")
            
        except Exception as e:
            st.error(f"Error processing document: {str(e)}")

    # Display chat interface
    if st.session_state.chain:
        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Chat input
        if prompt := st.chat_input("Ask about the document..."):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)

            try:
                # Get response from chain
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        response = st.session_state.chain({"question": prompt})
                        answer = response['answer']
                        
                        # Display the response
                        st.markdown(answer)
                        
                        # Add assistant message to chat history
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        
                        # Optionally display sources
                        if 'source_documents' in response:
                            with st.expander("View Sources"):
                                for i, doc in enumerate(response['source_documents']):
                                    st.write(f"Source {i+1}:", doc.page_content[:200] + "...")
                                    
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")

if __name__ == "__main__":
    main()