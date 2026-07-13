from langchain_core import output_parsers
from langchain_groq import ChatGroq
# Change this:
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os
import dotenv
import streamlit as st

dotenv.load_dotenv()

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# prompt
prompt= ChatPromptTemplate.from_messages(
    [
        ("system", "you're a helpful assisstant that resolves and answwers user's queries. Please respond accordingly"),
        ("user", "Question:{question}")
    ]
)

# Sreamlit 
st.title('Langchain test run')
input_text=st.text_input("Search topic")

# LLM 
llm = ChatGroq(model="llama-3.3-70b-versatile")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
output_parser=StrOutputParser()
chain =  prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question': input_text}))