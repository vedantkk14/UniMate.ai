# for Data Science

import os
import re
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatMessagePromptTemplate, MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import GoogleSearchAPIWrapper
from langchain_core.documents import Document
from dotenv import load_dotenv

# libraries for generating pdfs
import io
import json
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import HexColor
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

# libraries for OCR
import pytesseract
from pdf2image import convert_from_bytes
from PIL import Image
import cv2

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("⚠️ Please login to access this page")
    if st.button("Go to Login"):
        st.switch_page("login.py")
    st.stop()

if "user" not in st.session_state or st.session_state.user is None:
    st.error("❌ Session expired. Please login again.")
    if st.button("Go to Login", key="login_redirect"):
        st.switch_page("login.py")
    st.stop()

st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

def chat_model():
    llm = HuggingFaceEndpoint(
        repo_id='meta-llama/Meta-Llama-3.1-8B-Instruct',
        task='text-generation',
        temperature=0.6
    )
    return ChatHuggingFace(llm=llm)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model='sentence-transformers/all-MiniLM-L6-v2'
    )

def clean_text(text):
    """
    Improved text cleaning for ANN textbook - less aggressive.
    """
    if not text:
        return ""
    
    # 1. Fix broken words across lines (hyphenation) - BEFORE other cleaning
    text = re.sub(r'(\w)-\s*\n\s*(\w)', r'\1\2', text)
    
    # 2. Remove page numbers at start/end of lines
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    
    # 3. Normalize whitespace (but keep structure)
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    
    # Replace 3+ newlines with 2 (keep paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    # 4. Fix spacing around punctuation
    text = re.sub(r'\s+([.,;:!?])', r'\1', text)
    text = re.sub(r'([.,;:!?])([A-Za-z])', r'\1 \2', text)
    
    # 5. Remove common PDF artifacts (but keep actual content)
    # Remove isolated single characters on their own lines
    text = re.sub(r'\n\s*[a-z]\s*\n', '\n', text, flags=re.IGNORECASE)
    
    # 6. Strip whitespace from each line (but keep the lines)
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(line for line in lines if line)  # Remove empty lines
    
    # 7. Final whitespace cleanup
    text = text.strip()
    
    return text

def web_search_fallback(query: str) -> str:

    try:
        search = GoogleSearchAPIWrapper()
        search_results = search.run(query)

        if not search_results or search_results.strip == "":
            return "Could not find relevant information from the web search. Please try rephrasing your question"
        
        llm = chat_model()

        structured_prompt = PromptTemplate(
            template="""
            You are a Data Scientist (someone who is expert in Data Science). 
            A user asked: "{query}"

            Here are the web search results:
            {search_results}

            Please provide a clear, structured answer to the user's question based on these search results.
            Keep it concise and relevant to Data Science concepts.

        """,
        input_variables=['query', 'search_results'],
        validate_template=True
        )

        chain = structured_prompt | llm

        response = chain.invoke({
            'query' : query,
            'search_results' : search_results
        })
        return response.content
    
    except Exception as e:
        return f"An error occurred during web search fallback {str(e)}"
    
def load_and_create_vectordb(pdf_path='pdf/ds_tb.pdf',
                            vectordb_dir='vectordb/ds_faiss',
                            cache_dir='cache',
                            test_mode=False):

    embeddings = get_embeddings()

    if os.path.exists(vectordb_dir):
        st.info("📂 Loading existing vector database from disk...")
        try:
            vectordb = FAISS.load_local(
                vectordb_dir, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            st.success("✅ Vector database loaded successfully!")
            return vectordb
        except Exception as e:
            st.warning(f"⚠️ Could not load existing database: {str(e)}. Creating new one...")

        st.info("📄 No existing database found. Creating new vector database...")

        try:
            if not os.path.exists(pdf_path):
                st.error(f"❌ PDF file not found at: {pdf_path}")
                return None
            
            st.info(f"📖 Attempting standard text extraction from: {pdf_path}")
            loader = PyPDFLoader(pdf_path)
            pages = loader.load()

            raw_text = ""
            for page in pages:
                raw_text += page.page_content + "\n\n"

            if not raw_text.strip():
                st.warning("⚠️ Standard extraction failed (Scanned PDF detected). Switching to OCR... This will take longer.")

        except:
            pass