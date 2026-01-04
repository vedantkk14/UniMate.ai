from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.utilities import GoogleSearchAPIWrapper
from dotenv import load_dotenv
import streamlit as st
from pypdf import PdfReader
from datetime import datetime
import re
import database as db  
import time

# libraries for generating pdfs
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.colors import HexColor
import io

if "authenticated" not in st.session_state or not st.session_state.authenticated:
    st.warning("⚠️ Please login to access this page")
    st.info("👉 Click the button below to go to the login page")
    if st.button("Go to Login", type="primary"):
        st.switch_page("login.py")
    st.stop()

# Add this after authentication check in all 5th semester pages
st.set_page_config(page_title="5th Semester", page_icon="📘")

# Hide default Streamlit pages navigation
st.markdown("""
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
""", unsafe_allow_html=True)

# LLM Model (HuggingFace Endpoint)
def chat_model():
    llm = HuggingFaceEndpoint(
        repo_id="meta-llama/Meta-Llama-3.1-8B-Instruct",
        task="text-generation",
        temperature=0.8,
        max_new_tokens=512
    )
    return ChatHuggingFace(llm=llm)

# Embedding model
def get_embeddings():
    return HuggingFaceEmbeddings(
        model='sentence-transformers/all-MiniLM-L6-v2'
    )

def clean_text(text):
    """
    Clean text extracted from PDF by removing extra whitespace, 
    special characters, and normalizing formatting
    """
    if not text:
        return ""
    
    # Remove excessive whitespace and newlines
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,;:?!()\-\'\"@#$%&*+=/<>]', '', text)
    
    # Remove multiple consecutive punctuation marks
    text = re.sub(r'([.,;:!?]){2,}', r'\1', text)
    
    # Normalize quotes
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    
    # Remove page numbers (common pattern: Page X or X)
    text = re.sub(r'\bPage\s+\d+\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^\d+$', '', text, flags=re.MULTILINE)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text

def web_search_fallback(query: str) -> str:
    """
    Fallback function to search the web when information is not in the model's knowledge base
    """
    try:
        search = GoogleSearchAPIWrapper()
        search_results = search.run(query)

        llm = chat_model()

        structure_prompt = f"""You are a friendly and helpful AI assistant. 
        A user asked: "{query}"

        Here are the web search results:
        {search_results}

        Please provide a clear, friendly, and structured answer to the user's question based on these search results.
        Keep it conversational, concise, and helpful.
        """ 

        response = llm.invoke(structure_prompt)
        return response.content
    
    except Exception as e:
        return f"I apologize, but I couldn't find the relevant information right now. Please try again later or rephrase your question.\nError: {str(e)}"

def generate_chat_pdf(messages):
    """Generate a PDF from chat messages"""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=letter,
                    rightMargin=72, leftMargin=72,
                    topMargin=72, bottomMargin=18)
    
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
        leftIndent=20,
        rightIndent=20,
        spaceAfter=6,
        spaceBefore=12,
        backColor=HexColor('#dbeafe'),
        borderPadding=10,
        borderRadius=5
    )
    
    # Answer style (assistant)
    answer_style = ParagraphStyle(
        'Answer',
        parent=styles['Normal'],
        fontSize=10,
        textColor=HexColor('#374151'),
        leftIndent=20,
        rightIndent=20,
        spaceAfter=20,
        spaceBefore=6,
        backColor=HexColor('#f3f4f6'),
        borderPadding=10,
        borderRadius=5
    )
    
    # Label styles
    label_style = ParagraphStyle(
        'Label',
        parent=styles['Normal'],
        fontSize=9,
        textColor=HexColor('#6b7280'),
        spaceAfter=3
    )
    
    # Add title
    title = Paragraph("AI Assistant Conversation", title_style)
    elements.append(title)
    
    # Add timestamp
    timestamp = Paragraph(
        f"<i>Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</i>",
        label_style
    )
    elements.append(timestamp)
    elements.append(Spacer(1, 0.3*inch))
    
    # Add messages
    for i, message in enumerate(messages, 1):
        if message["role"] == "user":
            # Add Q label
            q_label = Paragraph(f"<b>Question {i//2 + 1}:</b>", label_style)
            elements.append(q_label)
            
            # Add question with HTML escaping
            question_text = message["content"].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            question = Paragraph(question_text, question_style)
            elements.append(question)
            
        else:  # assistant
            # Add A label
            a_label = Paragraph("<b>Answer:</b>", label_style)
            elements.append(a_label)
            
            # Add answer with HTML escaping
            answer_text = message["content"].replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
            answer = Paragraph(answer_text, answer_style)
            elements.append(answer)
            elements.append(Spacer(1, 0.2*inch))
    
    # Build PDF
    doc.build(elements)
    
    # Get the value of the BytesIO buffer
    pdf_data = buffer.getvalue()
    buffer.close()
    
    return pdf_data

# main functn that displays output: reads user query-> checks for ans in vectordb -> if no response -> checks in its own knowedge base -> if not found then performs google search
def get_general_response(user_question, chat_history):
    """
    Get response from the AI model using its own knowledge base
    """
    try:
        llm = chat_model()
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a friendly, helpful, and knowledgeable AI assistant. 
Your goal is to assist users with their questions in a conversational and approachable manner.

Guidelines:
- Be warm, friendly, and professional
- Provide clear and helpful answers based on your knowledge
- If you're unsure about something, be honest about it
- Keep responses concise but informative
- Use a conversational tone"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
        ])
        
        chain = (
            {
                "chat_history": lambda _: chat_history,
                "question": RunnablePassthrough()
            }
            | prompt
            | llm
        )
        
        response = chain.invoke(user_question)
        answer = response.content
        
        # Check if the model indicates it doesn't know
        uncertain_phrases = [
            "i don't know",
            "i'm not sure",
            "i cannot provide",
            "i don't have information",
            "beyond my knowledge",
            "i lack information"
        ]
        
        if any(phrase in answer.lower() for phrase in uncertain_phrases):
            # Use web search fallback
            answer = web_search_fallback(user_question)
            answer = "🌐 **Searched the web for you:**\n\n" + answer
        
        return answer
    
    except Exception as e:
        return f"I encountered an error processing your request. Please try again.\nError: {str(e)}"

def generate_quiz_from_history(chat_history):
    """
    Generates a quiz based on the provided chat history.
    """
    if not chat_history:
        return None

    # Convert list of messages to a single string for the prompt
    history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    llm = chat_model()

    quiz_prompt = f"""
    You are a teacher. Based strictly on the following conversation history, generate 5 multiple-choice questions (MCQs) to test the user's understanding of the topics discussed.
    
    Conversation History:
    {history_text}

    Rules:
    1. Generate exactly 5 questions.
    2. Provide 4 options (A, B, C, D) for each question.
    3. Indicate the correct answer clearly at the end of each question block.
    4. Do not include any introductory or concluding text. Use the exact format below.

    Format Example:
    Q1: What is normalization?
    A) Process of removing data
    B) Process of organizing data
    C) Process of deleting tables
    D) None of the above
    Answer: B
    
    Q2: ...
    """
    
    try:
        response = llm.invoke(quiz_prompt)
        return parse_quiz_content(response.content)
    except Exception as e:
        st.error(f"Error generating quiz: {e}")
        return None

def parse_quiz_content(text):
    """
    Parses the LLM output text into a structured list of dictionaries.
    """
    questions = []
    # Split by "Q" followed by a number and a colon/dot
    blocks = re.split(r'Q\d+[:.]', text)
    
    for block in blocks:
        if not block.strip():
            continue
            
        lines = [line.strip() for line in block.strip().split('\n') if line.strip()]
        if len(lines) < 5:
            continue
            
        question_text = lines[0]
        options = []
        correct_answer = None
        
        for line in lines[1:]:
            if line.upper().startswith(('A)', 'B)', 'C)', 'D)')):
                options.append(line)
            elif line.upper().startswith('ANSWER:'):
                correct_answer = line.split(':')[-1].strip().upper()
        
        if question_text and len(options) >= 4 and correct_answer:
            questions.append({
                "question": question_text,
                "options": options,
                "correct": correct_answer
            })
            
    return questions

# MAIN APP
def main():
    load_dotenv()
    st.set_page_config(page_title="5th Semester", layout="centered", page_icon="📘")

    # Hide default Streamlit navigation
    st.markdown("""
        <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "knowledge_base" not in st.session_state:
        st.session_state.knowledge_base = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    # for quiz questions
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None
    
    # Get user info
    user = st.session_state.user

    # Updated header with user name
    st.header(f"🤖 UniMate AI - Welcome, {user['name']}!")

    # Show current semester
    selected_semester = st.session_state.get('selected_semester', user['study_year'])
    st.info(f"📚 Current Learning Path: **{selected_semester}**")
        
    # Sidebar with PDF upload and controls
    with st.sidebar:

        # Navigation buttons
        if st.button("🏠 Back to Dashboard", use_container_width=True):
            st.switch_page("login.py")
        
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.switch_page("login.py")
        
        st.markdown("---")
        
        st.markdown("### 📚 5th Semester Subjects")
        
        if st.button("🏠 Home", use_container_width=True):
            st.switch_page("pages/5th_sem_home.py")
        
        if st.button("📘 AI", use_container_width=True):
            st.switch_page("pages/5th_sem_AI.py")
        
        if st.button("📗 CN", use_container_width=True):
            st.switch_page("pages/5th_sem_cn.py")
        
        if st.button("📙 DBMS", use_container_width=True):
            st.switch_page("pages/5th_sem_dbms.py")
        
        if st.button("📕 HCI", use_container_width=True):
            st.switch_page("pages/5th_sem_hci.py")
        
        if st.button("📓 WT", use_container_width=True):
            st.switch_page("pages/5th_sem_wt.py")
        
        st.markdown("---")
        
        # User Profile (your existing code)
        st.markdown("### 👤 Your Profile")
        user = st.session_state.user
        st.write(f"**{user['name']}**")
        st.write(f"📧 {user['email']}")
        st.write(f"🎓 5th Semester")
            
        st.markdown("---")

        # PDF Upload Section
        st.markdown("### 📁 Upload Document (Optional)")
        st.markdown("Upload a PDF to ask questions about its content")
        pdf = st.file_uploader("Upload your PDF:", type=["pdf"], label_visibility="collapsed")
        
        st.markdown("---")  

        st.markdown("### 📝 Knowledge Check")
        
        if st.button("Generate Quiz from Chat", use_container_width=True):
            if len(st.session_state.chat_history) < 2:
                st.warning("⚠️ Chat more with the bot first to generate a context-aware quiz!")
            else:
                with st.spinner("👩‍🏫 Analyzing conversation and crafting questions..."):
                    quiz = generate_quiz_from_history(st.session_state.chat_history[-10:])
                    if quiz:
                        st.session_state.quiz_data = quiz
                        st.success("✅ Quiz generated! Scroll down to take it.")
                    else:
                        st.error("❌ Failed to generate valid questions. Try again.")
        
        st.markdown("----")
        
        # if st.button("🔄 Reset Everything", use_container_width=True):
        #     st.session_state.messages = []
        #     st.session_state.knowledge_base = None
        #     st.session_state.chat_history = []
        #     st.rerun()
        
        st.markdown("### 📜 Export Chat")
        
        # PDF generation button
        if st.button("📥 Download Conversation as PDF", use_container_width=True):
            if st.session_state.messages:
                try:
                    pdf_data = generate_chat_pdf(st.session_state.messages)
                    
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

        st.markdown("---")
        # Chat Controls Section
        st.markdown("### 🎛️ Chat Controls")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.session_state.chat_history = []
            st.rerun()
            
    # Process PDF if uploaded
    if pdf is not None:
        if st.session_state.knowledge_base is None:
            with st.spinner("Processing PDF..."):
                
                raw_text = ""
                try:
                    # Attempt standard text extraction
                    pdf_reader = PdfReader(pdf)
                    for page in pdf_reader.pages:
                        extracted = page.extract_text()
                        if extracted:
                            raw_text += extracted + " "
                    
                    if not raw_text.strip():
                        st.warning("⚠️ Warning: No text extracted. Scanned PDFs/images are not supported.")

                except Exception as e:
                    st.error(f"Error during PDF processing: {str(e)}")
                
                # Clean and normalize the text
                text = clean_text(raw_text)
                
                if text:
                    # Split into chunks
                    text_splitter = CharacterTextSplitter(
                        separator="\n",
                        chunk_size=1000,
                        chunk_overlap=200,
                        length_function=len
                    )
                    chunks = text_splitter.split_text(text)
                    
                    # Generate embeddings + Vector DB
                    embeddings = get_embeddings()
                    st.session_state.knowledge_base = FAISS.from_texts(chunks, embeddings)
                    st.success("✅ PDF processed successfully! You can now ask questions about it.")
                else:
                    st.error("❌ Could not process PDF content.")
    
    # User input at the bottom
    user_question = st.chat_input("Ask anything..." if pdf is None else "Ask about the PDF or anything else...")
    
    if user_question:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        # Capture chat history before processing
        current_chat_history = st.session_state.chat_history.copy()
        
        # Determine if we should use PDF RAG or general knowledge
        if st.session_state.knowledge_base is not None:
            # PDF RAG Mode
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are a helpful and friendly AI assistant. 
Your task is to answer the user's question using the context from the uploaded PDF and the conversation history.

Guidelines:
- Provide detailed yet concise explanations
- Use clear and structured language
- If the answer is NOT in the PDF context, say: "I cannot find that information in the uploaded document. Let me search the web for you."
- Be conversational and helpful

--- 
PDF Context:
{context}"""),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])
            
            retriever = st.session_state.knowledge_base.as_retriever()
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
                response = rag_chain.invoke(user_question)
                answer = response.content
                
                # If not in PDF, fallback to web search
                if "cannot find that" in answer.lower() or "not in the document" in answer.lower():
                    web_answer = web_search_fallback(user_question)
                    answer = f"{answer}\n\n🌐 **Web Search Result:**\n{web_answer}"
        else:
            # General Knowledge Mode
            with st.spinner("Thinking..."):
                answer = get_general_response(user_question, current_chat_history)
        
        # Add to chat_history as LangChain message objects
        st.session_state.chat_history.append(HumanMessage(content=user_question))
        st.session_state.chat_history.append(AIMessage(content=answer))
        
        # Add assistant response to display messages
        st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # Rerun to update chat display
        st.rerun()
    
    # Display chat history
    if st.session_state.messages:
        st.markdown("---")
        chat_container = st.container()
        
        with chat_container:
            for message in st.session_state.messages:
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
    else:
        st.info("Ask me anything or upload a PDF to get started!")

if __name__ == "__main__":
    main()