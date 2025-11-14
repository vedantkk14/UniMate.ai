# for CN

import os
import re
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_community.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv

# pdf loader
import pytesseract
from pdf2image import convert_from_path

# libraries for generating pdfs
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.colors import HexColor
from datetime import datetime
import io

def chat_model():
    llm = HuggingFaceEndpoint(
        repo_id='openai/gpt-oss-20b',
        task='text-generation', 
        temperature=0.8
    )
    return ChatHuggingFace(llm=llm)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model='sentence-transformers/all-MiniLM-L6-v2'
    )

def clean_text(text):
    """Clean text while preserving the actual meaning"""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common OCR artifacts
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)  # Remove non-ASCII
    
    # Fix common OCR mistakes (customize based on your needs)
    text = text.replace('|', 'I')  # Common OCR error
    text = text.replace('0', 'O')  # In some contexts
    
    # Remove page numbers and headers/footers (adjust pattern as needed)
    text = re.sub(r'\n\d+\n', '\n', text)
    
    return text.strip()

def web_search_fallback(query: str) -> str:
    """
    This is a fallback function invoked when the model fails to generate output 
    from the given context from the database.
    """
    try:
        search = GoogleSearchAPIWrapper()
        search_results = search.run(query)

        # Handle empty search results
        if not search_results or search_results.strip() == "":
            return "I couldn't find relevant information from web search. Please try rephrasing your question."

        llm = chat_model()

        structured_prompt = PromptTemplate(
            template="""
            You are a Computer Networks (CN) expert assistant. 
            A user asked: "{query}"

            Here are the web search results:
            {search_results}

            Please provide a clear, structured answer to the user's question based on these search results.
            Keep it concise and relevant to Computer Networks concepts.
            """,
            input_variables=['query', 'search_results'],
            validate_template=True
        )

        chain = structured_prompt | llm

        response = chain.invoke({
            'query': query,
            'search_results': search_results
        })

        return response.content

    except Exception as e:
        return f"An error occurred during web search fallback: {str(e)}"
    
def extract_text_from_image_pdf(pdf_path):
    """Extract text from image-based PDF using OCR"""
    try:
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        st.info(f"🔍 Performing OCR on PDF: {pdf_path}")
        pages = convert_from_path(pdf_path, dpi=300)  # Higher DPI for better OCR
        text = ""

        for i, page in enumerate(pages):
            st.write(f"Processing page {i+1}/{len(pages)}...")
            page_text = pytesseract.image_to_string(page, lang='eng')
            
            # Clean the extracted text
            page_text = clean_text(page_text)
            text += page_text + "\n\n"

        st.success(f"✅ Extracted text from {len(pages)} pages")
        return text
    
    except Exception as e:
        st.error(f"❌ Error during OCR extraction: {str(e)}")
        return ""

def load_and_create_vectordb(pdf_path='pdfs/cn_tb.pdf', vectordb_dir='vectordb/cn_faiss'):
    """
    Load PDF, clean text, create chunks and build FAISS vector database
    """
    embeddings = get_embeddings()

    # Check if vector database already exists
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

    # Create a vector database if it doesn't exist
    st.info("🔨 Creating new vector database (this may take a moment)...")
    
    # Extract text from PDF
    extracted_text = extract_text_from_image_pdf(pdf_path)
    
    if not extracted_text:
        st.error("❌ No text extracted from PDF. Cannot create vector database.")
        return None
    
    # Create Document objects from the extracted text
    from langchain_core.documents import Document
    docs = [Document(page_content=extracted_text, metadata={"source": pdf_path})]
    
    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_documents(docs)
    
    st.info(f"📄 Created {len(chunks)} chunks from the document")
    
    # Create vector database
    vectordb = FAISS.from_documents(chunks, embeddings)
    
    # Save to disk for future use
    os.makedirs(vectordb_dir, exist_ok=True)
    vectordb.save_local(vectordb_dir)
    st.success(f"✅ Vector database created and saved to {vectordb_dir}")
    
    return vectordb