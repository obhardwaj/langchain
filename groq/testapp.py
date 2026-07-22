from langchain_core.prompts import ChatPromptTemplate
import streamlit as st
import os
import time
from langchain_groq import ChatGroq
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv


load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

if "vector" not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model='nomic-embed-text')
    st.session_state.loader = WebBaseLoader("https://docs.langchain.com/")
    st.session_state.docs=st.session_state.loader.load()
    st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=300)
    st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

st.title("ChatGroq Demo")
llm=ChatGroq(model='llama-3.3-70b-versatile') 
prompt = ChatPromptTemplate.from_template("""   
Answer the following question based only on the context provided.
think step by step before providing a detailed answer.
I will tip you $1000 if the user finds the answer useful.

<context>
{context}
</context>

Question: {input}
""")

document_chain=create_stuff_documents_chain(llm, prompt)
retriever=st.session_state.vectors.as_retriever()
retrieval_chain=create_retrieval_chain(retriever, document_chain)

user_question = st.text_input("Insert Prompt")

if user_question:
    start = time.process_time()
    response = retrieval_chain.invoke({"input": user_question})
    print("Response Time: ", time.process_time() - start)
    st.write(response['answer'])

    with st.expander("Document similarity search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write("-----------------------")