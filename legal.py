import os
import time
import tempfile
import requests
import urllib3
import streamlit as st
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain

# Suppress SSL warnings from NIC government server requests
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

st.title("⚖️ Indian Bare Acts Assistant")

llm = ChatGroq(model="llama-3.3-70b-versatile")

prompt = PromptTemplate.from_template(
    """
    You are an expert Indian Legal Assistant. Answer the question strictly based on the provided context.
    Cite relevant Section numbers, Sub-sections, and Act names wherever applicable.
    If the legal provision is not found in the context, explicitly state that it is not in the loaded documents.

    <context>
    {context}
    </context>

    Question: {input}
    """
)

# Known direct bitstream PDF links as fallbacks if browsing fails
FALLBACK_PDF_URLS = [
    "https://www.indiacode.nic.in/bitstream/123456789/20062/1/a202345.pdf",  # Bharatiya Nyaya Sanhita, 2023
    "https://www.indiacode.nic.in/bitstream/123456789/2187/2/A187209.pdf",    # Indian Contract Act, 1872
]

def load_acts(browse_url: str, max_acts: int = 2):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    }
    
    all_documents = []
    status_box = st.empty()
    pdf_urls_to_download = []

    # Step 1: Try scraping the India Code browse directory
    try:
        status_box.info("Fetching directory from India Code...")
        response = requests.get(browse_url, headers=headers, verify=False, timeout=15)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            
            act_links = []
            for a in soup.find_all("a", href=True):
                href = a['href']
                # Target act handles while skipping the main category handle and browse/search pages
                if "/handle/123456789/" in href and "/1362" not in href and "browse" not in href:
                    clean_href = href.split("?")[0]
                    full_url = "https://www.indiacode.nic.in" + clean_href
                    if full_url not in act_links:
                        act_links.append(full_url)

            # Extract bitstream PDF links from each Act page
            for act_url in act_links[:max_acts]:
                try:
                    res = requests.get(act_url, headers=headers, verify=False, timeout=15)
                    act_soup = BeautifulSoup(res.content, "html.parser")
                    for pdf_a in act_soup.find_all("a", href=True):
                        link = pdf_a['href']
                        # Strip parameters before checking extension
                        clean_link = link.split("?")[0]
                        if "/bitstream/" in link and clean_link.lower().endswith(".pdf"):
                            pdf_full_url = "https://www.indiacode.nic.in" + link
                            if pdf_full_url not in pdf_urls_to_download:
                                pdf_urls_to_download.append(pdf_full_url)
                            break
                except Exception as e:
                    st.warning(f"Failed to extract PDF link from {act_url}: {e}")
    except Exception as e:
        st.warning(f"Could not scrape browse directory: {e}")

    # Step 2: Use fallback URLs if directory crawling returned no links
    if not pdf_urls_to_download:
        status_box.warning("Directory crawling yielded no URLs. Using fallback Bare Acts URLs...")
        pdf_urls_to_download = FALLBACK_PDF_URLS[:max_acts]

    # Step 3: Download and parse the PDFs
    for idx, pdf_url in enumerate(pdf_urls_to_download, 1):
        try:
            status_box.info(f"Downloading Act {idx}/{len(pdf_urls_to_download)}: {pdf_url.split('/')[-1]}")
            pdf_res = requests.get(pdf_url, headers=headers, verify=False, timeout=30)
            
            if pdf_res.status_code == 200 and len(pdf_res.content) > 1000:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(pdf_res.content)
                    tmp_path = tmp_file.name

                loader = PyPDFLoader(tmp_path)
                docs = loader.load()
                
                # Tag metadata with actual source URL
                for doc in docs:
                    doc.metadata["source"] = pdf_url
                
                all_documents.extend(docs)
                os.remove(tmp_path)
            else:
                st.warning(f"HTTP {pdf_res.status_code} or empty response for {pdf_url}")
        except Exception as e:
            st.warning(f"Skipped downloading {pdf_url}: {e}")

    status_box.empty()
    return all_documents

def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        browse_url = "https://www.indiacode.nic.in/handle/123456789/1362/browse?type=shorttitle"
        docs = load_acts(browse_url, max_acts=2)
        
        if not docs:
            st.error("No documents were loaded. Please check your network connection or try loading local PDF files.")
            return False

        st.session_state.docs = docs
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        
        if not st.session_state.final_documents:
            st.error("Text splitting produced 0 chunks. Unable to create vector store.")
            return False

        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        return True

if st.button("Build Bare Acts Vector Store"):
    with st.spinner("Processing India Code Bare Acts & building FAISS index..."):
        if vector_embedding():
            st.success("Vector Store DB is Ready!")

# User Query Input
prompt1 = st.text_input("Ask your doubts from the Bare Acts:")

if prompt1:
    if "vectors" not in st.session_state:
        st.warning("Please click 'Build Bare Acts Vector Store' first!")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 4})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)
        
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        st.caption(f"Response Time: {time.process_time() - start:.2f} seconds")
        
        st.write(response['answer'])

        with st.expander("Document Similarity Search (Legal Context)"):
            for i, doc in enumerate(response["context"]):
                source_file = os.path.basename(doc.metadata.get("source", "Unknown"))
                page_num = doc.metadata.get("page", "N/A")
                st.markdown(f"**Chunk {i+1}** — `{source_file}` (Page {page_num})")
                st.write(doc.page_content)
                st.write("-------------------------------------")