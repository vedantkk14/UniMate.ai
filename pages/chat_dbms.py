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

import json
from thefuzz import fuzz # For matching similar questions


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

# get pyqs function
def get_unit_from_question_number(q_num_str):
    """
    Maps question number to Unit based on user logic:
    Q1, Q2 -> Unit 3
    Q3, Q4 -> Unit 4
    Q5, Q6 -> Unit 5
    Q7, Q8 -> Unit 6
    """
    # Extract the first digit found in the string (e.g., "Q1a" -> 1)
    match = re.search(r'\d+', str(q_num_str))
    if not match:
        return None
    
    q_num = int(match.group())
    
    if q_num in [1, 2]: return "Unit 3"
    if q_num in [3, 4]: return "Unit 4"
    if q_num in [5, 6]: return "Unit 5"
    if q_num in [7, 8]: return "Unit 6"
    
    return "Other"

def extract_questions_with_numbers(text):
    """
    Asks LLM to extract questions AND preserve their numbers for unit mapping.
    """
    llm = chat_model()
    
    # Modified prompt to keep numbers
    prompt = f"""
    You are an exam parser. extract questions from the following text.
    
    Raw Text: "{text}"
    
    Rules:
    1. Extract the Main Question Number (1, 2, 3...) and the Question Text.
    2. Format strictly as: "NUMBER :: QUESTION_TEXT"
    3. Ignore marks like "(10 marks)".
    4. If a question has sub-parts (a, b), treat them as part of the main number.
       Example: "1a :: Define SQL" or "1 :: Define SQL".
    
    Output Example:
    1 :: What is normalization?
    2 :: Explain 3NF.
    3 :: What is a Transaction?
    """
    
    try:
        response = llm.invoke(prompt)
        lines = [line.strip() for line in response.content.split('\n') if "::" in line]
        return lines
    except Exception as e:
        print(f"Error extracting: {e}")
        return []

def process_pyq_pdfs(folder_path=None, force_reprocess=False):
    # --- PATH SETUP ---
    if folder_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        folder_path = os.path.join(base_dir, 'pyq_pdfs', 'dbms')
        
    output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "PYQs/pyqs_master_dbms.json")

    # --- NEW LOGIC: CHECK IF JSON EXISTS ---
    if os.path.exists(output_file) and not force_reprocess:
        # st.info(f"📂 Loading cached PYQ analysis from {output_file}...") # Optional debug print
        try:
            with open(output_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            st.error(f"Error loading cached JSON: {e}. Reprocessing...")
            # If loading fails, we allow the code to continue to regeneration

    # --- BELOW IS THE EXISTING PROCESSING LOGIC ---
    if not os.path.exists(folder_path):
        return "FOLDER_MISSING"

    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    if not pdf_files:
        return "NO_FILES"

    # Initialize Data Structure for Units
    unit_database = {
        "Unit 3": [],
        "Unit 4": [],
        "Unit 5": [],
        "Unit 6": []
    }

    for pdf_file in pdf_files:
        loader = PyPDFLoader(os.path.join(folder_path, pdf_file))
        pages = loader.load()
        full_text = " ".join([p.page_content for p in pages])
        
        # Get raw lines "1 :: What is SQL"
        raw_lines = extract_questions_with_numbers(full_text[:3500]) 
        
        for line in raw_lines:
            try:
                # Split "1 :: Text" into "1" and "Text"
                q_num_str, q_text = line.split("::", 1)
                q_text = q_text.strip()
                
                # Determine Unit
                unit_name = get_unit_from_question_number(q_num_str)
                
                # Only process if it belongs to Units 3-6
                if unit_name and unit_name in unit_database:
                    
                    # --- FUZZY MATCHING INSIDE THE SPECIFIC UNIT ---
                    found = False
                    for existing in unit_database[unit_name]:
                        similarity = fuzz.token_sort_ratio(q_text.lower(), existing['question'].lower())
                        if similarity > 85:
                            existing['count'] += 1
                            # Keep the longer description
                            if len(q_text) > len(existing['question']):
                                existing['question'] = q_text
                            found = True
                            break
                    
                    if not found:
                        unit_database[unit_name].append({'question': q_text, 'count': 1})
                        
            except ValueError:
                continue # Skip lines that don't match format

    # Sort questions inside each unit by count
    for unit in unit_database:
        unit_database[unit].sort(key=lambda x: x['count'], reverse=True)

    # Save
    os.makedirs(os.path.dirname(output_file), exist_ok=True) # Ensure directory exists
    with open(output_file, 'w') as f:
        json.dump(unit_database, f)
        
    return unit_database

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
                        # UPDATED: Use dbms_ prefix
                        st.session_state.dbms_quiz_data = quiz
                        st.success("✅ Quiz generated! Scroll down to take it.")
                    else:
                        st.error("❌ Failed to generate valid questions. Try again.")
        
        st.markdown("----")

        st.markdown("### 📊 Exam Analysis")
        
        col1, col2 = st.columns([0.7, 0.3])
        
        with col1:
            if st.button("🧠 Analyze PYQs", use_container_width=True):
                with st.spinner("Loading analysis..."):
                    # force_reprocess defaults to False, so it loads JSON if available
                    result = process_pyq_pdfs() 
                    
                    if isinstance(result, str): # Handle error strings
                        if result == "FOLDER_MISSING":
                            st.error("❌ Folder 'pyq_pdfs' not found!")
                        elif result == "NO_FILES":
                            st.error("❌ No PDFs found in 'pyq_pdfs' folder.")
                    else:
                        st.success(f"✅ Loaded {sum(len(v) for v in result.values())} questions.")
                        # UPDATED: Use dbms_ prefix
                        st.session_state.dbms_show_pyq_results = True 
                        st.rerun()

        with col2:
            # Small refresh button to Force Update
            if st.button("🔄", help="Force re-scan of PDFs"):
                with st.spinner("Re-scanning PDFs (this takes time)..."):
                    result = process_pyq_pdfs(force_reprocess=True)
                    st.success("Analysis updated!")
                    # UPDATED: Use dbms_ prefix
                    st.session_state.dbms_show_pyq_results = True
                    st.rerun()
        
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
    
    # for quiz questions (UPDATED with dbms_ prefix)
    if "dbms_quiz_data" not in st.session_state:
        st.session_state.dbms_quiz_data = None
    # for pyq visibility (UPDATED with dbms_ prefix)
    if "dbms_show_pyq_results" not in st.session_state:
        st.session_state.dbms_show_pyq_results = False
    
# --- PYQ DASHBOARD SECTION ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_script_dir)
    json_path = os.path.join(project_root, "PYQs/pyqs_master_dbms.json")

    # COMBINED CHECK: File must exist AND the flag must be True (UPDATED with dbms_ prefix)
    if st.session_state.dbms_show_pyq_results and os.path.exists(json_path):
        
        # 1. Show the Close Button
        if st.button("❌ Close Analysis View", key="close_pyq"):
            # UPDATED: Use dbms_ prefix
            st.session_state.dbms_show_pyq_results = False
            st.rerun()
            
        # 2. Show the Content (Inside the same IF block!)
        with st.expander("📚 Frequent Questions (Unit-wise Analysis)", expanded=True):
            try:
                with open(json_path, 'r') as f:
                    pyq_data = json.load(f)
                
                if pyq_data:
                    # Create Tabs for each Unit
                    tab1, tab2, tab3, tab4 = st.tabs(["Unit 3", "Unit 4", "Unit 5", "Unit 6"])
                    
                    # Helper to render a list
                    def render_unit_questions(questions):
                        if not questions:
                            st.info("No questions found for this unit.")
                            return
                        
                        for item in questions:
                            c1, c2 = st.columns([0.15, 0.85])
                            with c1:
                                if item['count'] > 1:
                                    st.markdown(f":fire: **{item['count']}x**")
                                else:
                                    st.markdown(f"🔹 1x")
                            with c2:
                                st.write(item['question'])
                            st.markdown("---")

                    with tab1: render_unit_questions(pyq_data.get("Unit 3", []))
                    with tab2: render_unit_questions(pyq_data.get("Unit 4", []))
                    with tab3: render_unit_questions(pyq_data.get("Unit 5", []))
                    with tab4: render_unit_questions(pyq_data.get("Unit 6", []))
                        
                else:
                    st.warning("JSON is empty. Try running analysis again.")
            except Exception as e:
                st.error(f"Error loading PYQ data: {e}")

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

    # (UPDATED with dbms_ prefix)
    if st.session_state.dbms_quiz_data:
        st.markdown("---")
        with st.expander("📝 Take the Quiz", expanded=True):
            st.subheader("Test Your Understanding")
            
            with st.form("quiz_form"):
                score = 0
                user_answers = {}
                
                # (UPDATED with dbms_ prefix)
                for i, q in enumerate(st.session_state.dbms_quiz_data):
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
                    # (UPDATED with dbms_ prefix)
                    for i, q in enumerate(st.session_state.dbms_quiz_data):
                        user_choice = user_answers.get(i)
                        # Extract the letter (A, B, C, D) from the user's choice string
                        user_letter = user_choice.split(')')[0] if user_choice else None
                        
                        if user_letter == q['correct']:
                            correct_count += 1
                            st.success(f"Q{i+1}: Correct!")
                        else:
                            st.error(f"Q{i+1}: Incorrect. The correct answer was {q['correct']}.")
                    
                    # (UPDATED with dbms_ prefix)
                    percentage = (correct_count / len(st.session_state.dbms_quiz_data)) * 100
                    st.metric(label="Final Score", value=f"{percentage}%", delta=f"{correct_count}/{len(st.session_state.dbms_quiz_data)} Correct")
                    
                    if percentage == 100:
                        st.balloons()

if __name__ == "__main__":
    main()