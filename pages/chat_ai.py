# AI page

import os
import re
import streamlit as st
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnableSequence, RunnablePassthrough
from langchain_community.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv

# libraries for generating pdfs
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, KeepTogether
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.colors import HexColor
from datetime import datetime
import io


def chat_model():
    llm = HuggingFaceEndpoint(
        repo_id='openai/gpt-oss-20b', 
        task='conversational',
        temperature=0.8
    )
    return ChatHuggingFace(llm=llm)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model='sentence-transformers/all-MiniLM-L6-v2'
    )

def clean_text(text):
    """Clean text while preserving the actual meaning"""

    pass
    
def web_search_fallback(query: str) -> str:
    """
    This is a fallback function invoked when the model fails to generate output from the given context from the database.
    """
    try:
        search = GoogleSearchAPIWrapper()
        search_results = search.run(query)

        llm = chat_model()

        structure_prompt = f"""You are a AI expert assistant. 
        A user asked: "{query}"

        Here are the web search results:
        {search_results}

        Please provide a clear, structured answer to the user's question based on these search results.
        Keep it concise and relevant to core AI concepts.
        """ 

        response = llm.invoke(structure_prompt)
        return response.content

    except Exception as e:
        return f"Apologies, I couldn't find the relevant information right noe. Please try again later.\nError: {str(e)}"

def load_and_create_vectordb(vectordb_path='vectordb/ai_faiss'):
    """
    Load PDF, clean text, create chunks and build FAISS vector database
    """
    embeddings = get_embeddings()

    # checking if vector database already exists,
    if os.path.exists(vectordb_path):
        st.info("📂 Loading existing vector database from disk...")
        try:
            vectordb = FAISS.load_local(
                vectordb_path,
                embeddings,
                allow_dangerous_deserialization=True
            )
            st.success("✅ Vector database loaded successfully!")
            return vectordb
        except Exception as e:
            st.warning(f"⚠️ Could not load existing database: {str(e)}. Creating new one...")


    # creating a vector database if it dosen't exists
    st.info("🔨 Creating new vector database (this may take a moment)...")
    loader = PyPDFLoader(file_path='pdfs/ai_tb.pdf')
    docs = loader.load()

    # for doc in docs:
    #     doc.page_content = clean_text(doc.page_content)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250
    )
    chunks = splitter.split_documents(docs)

    vectordb = FAISS.from_documents(chunks, embeddings)

    # saving the vectordb locally
    os.makedirs(os.path.dirname(vectordb_path), exist_ok=True)
    vectordb.save_local(vectordb_path)
    st.success("✅ Vector database created and saved to disk!")

    return vectordb


def clean_and_format_pdf_text(text):
    """Clean markdown and special characters, convert to HTML"""
    # Replace HTML special characters first
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # Convert markdown bold to HTML bold
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    
    # Convert markdown italic to HTML italic
    text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', text)
    
    # Convert backticks to monospace
    text = re.sub(r'`(.+?)`', r'<font face="Courier">\1</font>', text)
    
    # Convert line breaks to HTML breaks
    text = text.replace('\n', '<br/>')
    
    # Handle bullet points
    text = re.sub(r'^\s*[-*]\s+', '• ', text, flags=re.MULTILINE)
    
    return text

def generate_chat_pdf(messages):
    """Generate a PDF from chat messages"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer, 
        pagesize=letter,
        rightMargin=72, 
        leftMargin=72,
        topMargin=72, 
        bottomMargin=72
    )
    
    # Container for the 'Flowable' objects
    elements = []
    
    # Define styles
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=HexColor('#1e40af'),
        spaceAfter=30,
        alignment=TA_LEFT
    )
    
    # Question style (user)
    question_style = ParagraphStyle(
        'Question',
        parent=styles['Normal'],
        fontSize=11,
        textColor=HexColor('#1f2937'),
        leftIndent=10,
        rightIndent=10,
        spaceAfter=12,
        spaceBefore=6,
        backColor=HexColor('#dbeafe'),
        leading=14  # Line spacing
    )
    
    # Answer style (assistant)
    answer_style = ParagraphStyle(
        'Answer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor('#374151'),
        leftIndent=10,
        rightIndent=10,
        spaceAfter=12,
        spaceBefore=6,
        backColor=HexColor('#f3f4f6'),
        leading=13  # Line spacing
    )
    
    # Label styles
    label_style = ParagraphStyle(
        'Label',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor('#6b7280'),
        spaceAfter=4,
        spaceBefore=8
    )
    
    # Add title
    title = Paragraph("AI Chatbot Conversation", title_style)
    elements.append(title)
    
    # Add timestamp
    timestamp = Paragraph(
        f"<i>Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</i>",
        label_style
    )
    elements.append(timestamp)
    elements.append(Spacer(1, 0.3*inch))
    
    # Add messages
    question_count = 0
    for message in messages:
        if message["role"] == "user":
            question_count += 1
            
            # Create question block
            q_label = Paragraph(f"<b>Question {question_count}:</b>", label_style)
            question_text = clean_and_format_pdf_text(message["content"])
            question = Paragraph(question_text, question_style)
            
            # Keep label and question together
            elements.append(KeepTogether([q_label, question]))
            
        else:  # assistant
            # Create answer block
            a_label = Paragraph("<b>Answer:</b>", label_style)
            answer_text = clean_and_format_pdf_text(message["content"])
            answer = Paragraph(answer_text, answer_style)
            
            # Keep label and answer together
            elements.append(KeepTogether([a_label, answer]))
            elements.append(Spacer(1, 0.15*inch))
    
    # Build PDF
    doc.build(elements)
    
    # Get the value of the BytesIO buffer
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

def main():
    load_dotenv()
    st.set_page_config(
        page_title="AI Chatbot ",
        page_icon="🤖",
        layout="centered"
    )
    st.title("📘 AI Chatbot")

    with st.sidebar:

        st.markdown("### 🎛️ Chat Controls")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.ai_messages = []
            st.session_state.ai_chat_history = []
            st.rerun()

        st.markdown("----")

        st.markdown("### 📜 Generate PDF")

        # PDF generation button
        if st.button("📥 Download Conversation as PDF", use_container_width=True):
            if "ai_messages" in st.session_state and st.session_state.ai_messages:
                try:
                    pdf_data = generate_chat_pdf(st.session_state.ai_messages)
                    
                    # Create download button
                    st.download_button(
                        label="💾 Click to Download PDF",
                        data=pdf_data,
                        file_name=f"ai_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ PDF generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
            else:
                st.warning("⚠️ No conversation to export. Start chatting first!")
        
        st.markdown("----")

        # Button to recreate vector database
        if st.button("🔄 Rebuild Vector Database", use_container_width=True):
            vectordb_path = 'vectordb/ai_faiss'
            try:
                # Remove existing database
                if os.path.exists(vectordb_path):
                    import shutil
                    shutil.rmtree(vectordb_path)
                    st.info("🗑️ Removed old database...")
                
                # Recreate database
                with st.spinner("Rebuilding vector database..."):
                    st.session_state.ai_vectordb = load_and_create_vectordb(vectordb_path)
                st.success("✅ Database rebuilt successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error rebuilding database: {str(e)}")

    # session states and their uses
    # ai_messages->UI Display->Show chat bubbles, export PDF
    if "ai_messages" not in st.session_state:
        st.session_state.ai_messages = []
    # ai_chat_history->LLM Contextlist->Give memory to the model
    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []
    # ai_vectordb->RAG Retrieval->FAISS object->Avoid reloading PDF every time
    if "ai_vectordb" not in st.session_state:
        with st.spinner("Loading AI knowledge base..."):
            st.session_state.ai_vectordb = load_and_create_vectordb()
        st.success("Knowledge base loaded!")

    # user input 
    user_input = st.chat_input("What's on your mind?")

    if user_input:
        # add user messg to messages
        st.session_state.ai_messages.append({
            "role" : "user",
            "content" : user_input
        })

        # capture chat history before building the chain
        current_chat_history = st.session_state.ai_chat_history.copy()

        # check if its casual greeting
        casual_greeting = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']

        is_casual = any(greeting in user_input.lower().strip() for greeting in casual_greeting)

        if is_casual and len(user_input.split()) <= 3:
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a friendly AI expert assistant. Respond warmly to greetings."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])
            
            chain = (
                {
                    "chat_history" : lambda _: current_chat_history,
                    "question" : RunnablePassthrough()
                }
                | prompt
                | chat_model()
            )

            with st.spinner("Thinking..."):
                response = chain.invoke(user_input)
                answer = response.content
        else:
            # use RAG for AI-related questions
            retriever = st.session_state.ai_vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k":3}
            )

            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert AI assistant who has mastered Database Management Systems (AI).
Answer the question using the context extracted from the AI textbook and the previous conversation.

Guidelines for your response:
- Provide a **detailed yet concise** explanation.
- Use **clear, simple, and structured** language.
- If the answer is **not present** in the context, respond only with: "I don't have enough information in the knowledge base to answer this question."
- Do not add unrelated information beyond what is asked.

--- 
AI Context:
{context}"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ]) 

            rag_chain = (
                {
                    "context": retriever | (lambda docs: "\n\n".join(d.page_content for d in docs)),
                    "chat_history": lambda _: current_chat_history,
                    "question": RunnablePassthrough()
                }
                | prompt
                | chat_model()
            )

            with st.spinner("Thinking..."):
                response = rag_chain.invoke(user_input)
                answer = response.content

            if "FALLBACK_TRIGGER" in answer or "don't have enough information" in answer.lower():
                with st.spinner("🌐 Searching the web for more information..."):
                    answer = web_search_fallback(user_input)

                    answer = f"ℹ️ *From web search:*\n\n{answer}"

        st.session_state.ai_chat_history.append(HumanMessage(content=user_input))
        st.session_state.ai_chat_history.append(AIMessage(content=answer))
        st.session_state.ai_messages.append({"role" : "assistant", "content":answer})

        st.rerun()

        # Display chat history with custom UI
    st.markdown("---")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.ai_messages:
            if message["role"] == "user":
                st.markdown(f"""
                <div style='text-align: right; margin: 10px 0;'>
                    <div style='color: #666; font-size: 12px; margin-bottom: 5px;'>👤 You</div>
                    <span style='background-color: #007bff; color: white; padding: 10px 15px; 
                    border-radius: 15px; display: inline-block; max-width: 70%;'>
                        {message["content"]}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='text-align: left; margin: 10px 0;'>
                    <div style='color: #666; font-size: 12px; margin-bottom: 5px;'>🤖 Assistant</div>
                    <span style='background-color: #f1f1f1; color: black; padding: 10px 15px; 
                    border-radius: 15px; display: inline-block; max-width: 70%;'>
                        {message["content"]}
                    </span>
                </div>
                """, unsafe_allow_html=True)


            

if __name__ == "__main__":
    main()