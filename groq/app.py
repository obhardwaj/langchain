import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv


load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

st.title("CHATGROQ with LLAMA3")

llm = ChatGroq(model="llama-3.1-8b-instant")

prompt= PromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    <context>
    Questions:{input}
    """
)

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings()
        st.session_state.loader=PyPDFDirectoryLoader("./census")
        st.session_state.docs=st.session_state.loader.load()
        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

prompt1 = st.text_input("Ask your doubts from the Documents")

if st.button("Documents Embeddings"):
    vector_embedding()
    st.write("Vector Store DB is Ready!!!")

if prompt1:
    if "vectors" in st.session_state:
        document_chain=create_stuff_documents_chain(llm, prompt)
        retriever=st.session_state.vectors.as_retriever()
        retrieval_chain=create_retrieval_chain(retriever, document_chain)
        start=time.process_time()
        response=retrieval_chain.invoke({'input': prompt1})
        print("Response Time :", time.process_time()-start())
        st.write(response['answer'])
    


        with st.expander("Document Similarity Search"):
            for i, doc in enumerate(response["context"]):
                st.write(doc.page_content)
                st.write("-------------------------------------")