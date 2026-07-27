import os
import time
import streamlit as st
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_objectbox.vectorstores import ObjectBox
from langchain_community.document_loaders import PyPDFDirectoryLoader

# Load environment variables from .env
load_dotenv()

# Initialize LLM (Groq automatically reads GROQ_API_KEY from environment)
llm = ChatGroq(model_name="llama-3.3-70b-versatile")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question.
    <context>
    {context}
    </context>
    Questions:{input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        st.session_state.loader = PyPDFDirectoryLoader("./census")
        st.session_state.docs = st.session_state.loader.load()
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])
        st.session_state.vectors = ObjectBox.from_documents(
            st.session_state.final_documents, 
            st.session_state.embeddings, 
            embedding_dimensions=768
        )

st.title("ObjectBox Document Q&A")

input_prompt = st.text_input("Ask your Question from the Document")

if st.button("Documents Embedding"):
    vector_embedding()
    st.success("ObjectBox DB is ready!!")

if input_prompt:
    # Ensure vectors are loaded before trying to construct retriever
    if "vectors" not in st.session_state or st.session_state.vectors is None:
        st.warning("Please click the 'Documents Embedding' button first to process your files.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever()
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': input_prompt})
        print("Response Time:", time.process_time() - start)
        
        st.write(response['answer'])

        with st.expander("Document Similarity Search"):
            for doc in response["context"]:
                st.write(doc.page_content)
                st.write("------------------------")