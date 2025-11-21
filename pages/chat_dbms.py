# DBMS page

# libraries required for langchain
import os
import re
from datetime import datetime
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from langchain_community.utilities import GoogleSearchAPIWrapper
import streamlit as st

# libraries for generating pdfs
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
from reportlab.lib.enums import TA_LEFT, TA_RIGHT
from reportlab.lib.colors import HexColor
import io


def chat_model():
    llm = HuggingFaceEndpoint(
        repo_id='Qwen/Qwen2.5-7B-Instruct',
        task='text-generation',
        temperature=0.8,
        max_new_tokens=512
    )
    return ChatHuggingFace(llm=llm)

def get_embeddings():
    return HuggingFaceEmbeddings(
        model='sentence-transformers/all-MiniLM-L6-v2'
    )

def clean_text(text):
    """Clean text while preserving SQL-relevant characters"""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s.,!?:;\-=<>()\'"*/+]', '', text)
    text = re.sub(r'(.)\1{3,}', r'\1', text)
    return text.strip()

def web_search_fallback(query: str) -> str:
    """
    This is a Fallback function invoked when the model fails to generate output from the given context of the database.
    """
    try:
        search = GoogleSearchAPIWrapper()
        search_results = search.run(query)

        llm = chat_model()

        structure_prompt = f"""You are a DBMS expert assistant. 
        A user asked: "{query}"

        Here are the web search results:
        {search_results}

        Please provide a clear, structured answer to the user's question based on these search results.
        Keep it concise and relevant to DBMS concepts.
        """ 

        response = llm.invoke(structure_prompt)
        return response.content
    
    except Exception as e:
        return f"Apologies, I couldn't find the relevant information right now. Please try again later.\nError: {str(e)}"

def load_and_create_vectordb(vectordb_path='vectordb/dbms_faiss'):
    """Load PDF, clean text, create chunks, and build FAISS vector database"""
    embeddings = get_embeddings()
    
    # Check if vector database already exists
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
    
    # Create new vector database if it doesn't exist
    st.info("🔨 Creating new vector database (this may take a moment)...")
    loader = PyPDFLoader(file_path='pdfs/dbms_tb.pdf')
    docs = loader.load()
    
    for doc in docs:
        doc.page_content = clean_text(doc.page_content)
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=250
    )
    chunks = splitter.split_documents(docs)
    
    vectordb = FAISS.from_documents(chunks, embeddings)
    
    # Save the vector database locally
    os.makedirs(os.path.dirname(vectordb_path), exist_ok=True)
    vectordb.save_local(vectordb_path)
    st.success("✅ Vector database created and saved to disk!")
    
    return vectordb

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
    title = Paragraph("DBMS Chatbot Conversation", title_style)
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

def generate_quiz_from_history(chat_history):
    """
    Generates a quiz based on the provided chat history.
    """
    if not chat_history:
        return None

    # Convert list of messages to a single string for the prompt
    history_text = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    llm = chat_model()

    # Strict prompt to ensure consistent formatting for parsing
    quiz_prompt = f"""
    You are a teacher. Based strictly on the following conversation history about DBMS, generate 5 multiple-choice questions (MCQs) to test the user's understanding of the topics discussed.
    
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

def main():
    load_dotenv()
    st.set_page_config(
        page_title="DBMS Chatbot ",
        page_icon="🤖",
        layout="centered"
    )
    st.title("📘 DBMS Chatbot")

    with st.sidebar:

        st.markdown("### 📝 Knowledge Check")
        
        if st.button("Generate Quiz from Chat", use_container_width=True):
            if len(st.session_state.dbms_chat_history) < 2:
                st.warning("⚠️ Chat more with the bot first to generate a context-aware quiz!")
            else:
                with st.spinner("👩‍🏫 Analyzing conversation and crafting questions..."):
                    quiz = generate_quiz_from_history(st.session_state.dbms_chat_history[-10:])
                    if quiz:
                        st.session_state.quiz_data = quiz
                        st.success("✅ Quiz generated! Scroll down to take it.")
                    else:
                        st.error("❌ Failed to generate valid questions. Try again.")
        
        st.markdown("----")

        st.markdown("### 📜 Generate PDF")
        
        # PDF generation button
        if st.button("📥 Download Conversation as PDF", use_container_width=True):
            if "dbms_messages" in st.session_state and st.session_state.dbms_messages:
                try:
                    pdf_data = generate_chat_pdf(st.session_state.dbms_messages)
                    
                    # Create download button
                    st.download_button(
                        label="💾 Click to Download PDF",
                        data=pdf_data,
                        file_name=f"dbms_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
                    st.success("✅ PDF generated successfully!")
                except Exception as e:
                    st.error(f"❌ Error generating PDF: {str(e)}")
            else:
                st.warning("⚠️ No conversation to export. Start chatting first!")

        st.markdown("----")
        
        st.markdown("### 🎛️ Chat Controls")
        
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.dbms_messages = []
            st.session_state.dbms_chat_history = []
            st.rerun()

        st.markdown("----")
        
        # st.markdown("### 🗄️ Database Management")
        
        # # Button to recreate vector database
        # if st.button("🔄 Rebuild Vector Database", use_container_width=True):
        #     vectordb_path = 'vectordb/dbms_faiss'
        #     try:
        #         # Remove existing database
        #         if os.path.exists(vectordb_path):
        #             import shutil
        #             shutil.rmtree(vectordb_path)
        #             st.info("🗑️ Removed old database...")
                
        #         # Recreate database
        #         with st.spinner("Rebuilding vector database..."):
        #             st.session_state.dbms_vectordb = load_and_create_vectordb(vectordb_path)
        #         st.success("✅ Database rebuilt successfully!")
        #         st.rerun()
        #     except Exception as e:
        #         st.error(f"❌ Error rebuilding database: {str(e)}")

    # Initialize session state with page-specific keys
    if "dbms_messages" not in st.session_state:
        st.session_state.dbms_messages = []
    if "dbms_chat_history" not in st.session_state:
        st.session_state.dbms_chat_history = []
    if "dbms_vectordb" not in st.session_state:
        with st.spinner("Loading DBMS knowledge base..."):
            st.session_state.dbms_vectordb = load_and_create_vectordb()
        st.success("Knowledge base loaded!")
    # for quiz questions
    if "quiz_data" not in st.session_state:
        st.session_state.quiz_data = None

    # User input at the bottom (placed before chat display)
    user_input = st.chat_input("Ask anything about DBMS")

    if user_input:
        # Add user message to messages
        st.session_state.dbms_messages.append({"role": "user", "content": user_input})
        
        # Capture chat history before building the chain
        current_chat_history = st.session_state.dbms_chat_history.copy()
        
        # Check if it's a casual greeting
        casual_greetings = ['hi', 'hello', 'hey', 'greetings', 'good morning', 'good afternoon', 'good evening']
        is_casual = any(greeting in user_input.lower().strip() for greeting in casual_greetings)
        
        if is_casual and len(user_input.split()) <= 3:
            # Handle casual conversation without RAG
            prompt = ChatPromptTemplate.from_messages([
                ("system", "You are a friendly DBMS expert assistant. Respond warmly to greetings."),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}")
            ])
            
            chain = (
                {
                    "chat_history": lambda _: current_chat_history,
                    "question": RunnablePassthrough()
                }
                | prompt
                | chat_model()
            )
            
            with st.spinner("Thinking..."):
                response = chain.invoke(user_input)
                answer = response.content
        else:
            # Use RAG for DBMS-related questions
            retriever = st.session_state.dbms_vectordb.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            )
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", """You are an expert AI assistant who has mastered Database Management Systems (DBMS).
Answer the question using the context extracted from the DBMS textbook and the previous conversation.

Guidelines for your response:
- Provide a **detailed yet concise** explanation.
- Use **clear, simple, and structured** language.
- If the answer is **not present** in the context, respond only with: "I don't have enough information in the knowledge base to answer this question."
- Do not add unrelated information beyond what is asked.

--- 
DBMS Context:
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

                    # Adding a note that this came from the web search...
                    answer = f"ℹ️ *From web search:*\n\n{answer} "
        
        # Add to chat_history as LangChain message objects
        st.session_state.dbms_chat_history.append(HumanMessage(content=user_input))
        st.session_state.dbms_chat_history.append(AIMessage(content=answer))
        
        # Add assistant response to display messages
        st.session_state.dbms_messages.append({"role": "assistant", "content": answer})
        
        # Rerun to update chat display
        st.rerun()

    # Display chat history with custom UI
    st.markdown("---")
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.dbms_messages:
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

    if st.session_state.quiz_data:
        st.markdown("---")
        with st.expander("📝 Take the Quiz", expanded=True):
            st.subheader("Test Your Understanding")
            
            with st.form("quiz_form"):
                score = 0
                user_answers = {}
                
                for i, q in enumerate(st.session_state.quiz_data):
                    st.markdown(f"**{i+1}. {q['question']}**")
                    # Radio button for options
                    user_choice = st.radio(
                        "Select an answer:", 
                        q['options'], 
                        key=f"q_{i}",
                        index=None # No default selection
                    )
                    user_answers[i] = user_choice
                    st.markdown("---")
                
                submitted = st.form_submit_button("Submit Answers")
                
                if submitted:
                    correct_count = 0
                    for i, q in enumerate(st.session_state.quiz_data):
                        user_choice = user_answers.get(i)
                        # Extract the letter (A, B, C, D) from the user's choice string
                        user_letter = user_choice.split(')')[0] if user_choice else None
                        
                        if user_letter == q['correct']:
                            correct_count += 1
                            st.success(f"Q{i+1}: Correct!")
                        else:
                            st.error(f"Q{i+1}: Incorrect. The correct answer was {q['correct']}.")
                    
                    percentage = (correct_count / len(st.session_state.quiz_data)) * 100
                    st.metric(label="Final Score", value=f"{percentage}%", delta=f"{correct_count}/{len(st.session_state.quiz_data)} Correct")
                    
                    if percentage == 100:
                        st.balloons()

if __name__ == "__main__":
    main()