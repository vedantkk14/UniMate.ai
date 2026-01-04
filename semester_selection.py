"""
Semester Selection Page for UniMate AI
Displays after successful login
"""

import streamlit as st

def show_semester_selection():
    """Show semester selection dashboard after login"""

    # Hide default Streamlit navigation
    st.markdown("""
        <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Custom CSS for beautiful dashboard
    st.markdown("""
        <style>
        .welcome-header {
            text-align: center;
            color: #1f77b4;
            font-size: 2.5em;
            font-weight: bold;
            margin-bottom: 10px;
            animation: fadeIn 1s;
        }
        .welcome-subheader {
            text-align: center;
            color: #666;
            font-size: 1.2em;
            margin-bottom: 30px;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(-20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .semester-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 20px;
            border-radius: 15px;
            color: white;
            text-align: center;
            margin: 10px 0;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .action-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            border: 2px solid #e0e0e0;
            margin: 10px 0;
            transition: transform 0.3s, box-shadow 0.3s;
        }
        .action-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        }
        .profile-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 20px;
            border-radius: 10px;
            color: white;
            margin-bottom: 20px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Get user data from session
    user = st.session_state.user
    
    # Welcome header
    st.markdown(f'<h1 class="welcome-header">👋 Welcome, {user["name"]}!</h1>', unsafe_allow_html=True)
    st.markdown('<p class="welcome-subheader">Ready to continue your learning journey?</p>', unsafe_allow_html=True)
    
    # Sidebar with user profile
    with st.sidebar:
        st.markdown("### 👤 Your Profile")
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.write(f"**📧 Email:** {user['email']}")
        st.write(f"**🏫 College:** {user['college']}")
        st.write(f"**📚 Semester:** {user['study_year']}")
        
        if user.get('department') and user['department'] != 'Not specified':
            st.write(f"**🎓 Department:** {user['department']}")
        
        st.markdown("---")
        
        # Profile actions
        st.markdown("### ⚙️ Account Settings")
        
        if st.button("✏️ Edit Profile", use_container_width=True):
            st.info("🚧 Profile editing feature coming soon!")
        
        if st.button("🔒 Change Password", use_container_width=True):
            st.info("🚧 Password change feature coming soon!")
        
        st.markdown("---")
        
        # Logout button
        if st.button("🚪 Logout", use_container_width=True, type="secondary"):
            # Clear session state
            st.session_state.authenticated = False
            st.session_state.user = None
            if 'messages' in st.session_state:
                st.session_state.messages = []
            if 'chat_history' in st.session_state:
                st.session_state.chat_history = []
            
            st.success("✅ Logged out successfully!")
            st.rerun()
    
    # Main content area
    st.markdown("---")
    
    # Semester selection section
    st.markdown("### 📚 Select Your Learning Path")
    st.write("Choose the semester you want to explore and access study materials, chatbot, and more!")
    
    # Create columns for semester buttons
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📙 5th Semester", use_container_width=True, type="primary", key="sem_5"):
            st.session_state.selected_semester = "5th Semester"
            st.switch_page("pages/sem5_home.py")
    
    with col2:
        if st.button("📕 6th Semester", use_container_width=True, type="primary", key="sem_6"):
            st.session_state.selected_semester = "6th Semester"
            st.switch_page("pages/sem6_home.py")
            # st.info("🚧 Content for 6th semester coming soon!")
    
    # Additional semesters
    col5, col6, col7, col8 = st.columns(4)
    
    with col5:
        if st.button("📓 7th Semester", use_container_width=True, key="sem_7"):
            st.info("🚧 Content for 7th semester coming soon!")
    
    with col6:
        if st.button("📔 8th Semester", use_container_width=True, key="sem_8"):
            st.info("🚧 Content for 8th semester coming soon!")
    
    st.markdown("---")
    
    # Quick Actions Section
    st.markdown("### 🎯 Quick Actions")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.container():
            st.markdown("#### 💬 AI Chatbot")
            st.write("Get instant answers to your academic questions")
            if st.button("Start Chatting", key="chat_btn", use_container_width=True):
                st.session_state.selected_semester = user['study_year']
                try:
                    st.switch_page("pages/home.py")
                except:
                    # st.error("❌ Home page not found at 'pages/home.py'")
                    st.info("🚧 AI chatbot coming soon...")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        with st.container():
            st.markdown("#### 📝 Practice Quiz")
            st.write("Test your knowledge with AI-generated quizzes")
            if st.button("Take Quiz", key="quiz_btn", use_container_width=True):
                # st.info("💡 Please select a semester first to access quizzes!")
                st.info("🚧 Quiz section to be added soon...")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        with st.container():
            st.markdown("#### 📊 Study Progress")
            st.write("Track your learning journey and achievements")
            if st.button("View Progress", key="progress_btn", use_container_width=True):
                # st.info("🚧 Progress tracking feature coming soon!")
                st.info("🚧 Progress section to be added soon...")
            st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features showcase
    st.markdown("### ✨ What You Can Do with UniMate AI")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🤖 Smart AI Assistant**
        - Ask questions about any topic
        - Get explanations in simple terms
        - 24/7 availability
        
        **📄 PDF Document Analysis**
        - Upload study materials
        - Ask questions about PDFs
        - Extract key information
        
        **🧠 Personalized Learning**
        - Tailored to your semester
        - Adaptive difficulty
        - Topic-focused content
        """)
    
    with col2:
        st.markdown("""
        **📝 Interactive Quizzes**
        - Auto-generated from conversations
        - Multiple choice questions
        - Instant feedback
        
        **💾 Export Features**
        - Download chat history as PDF
        - Save important conversations
        - Share with friends
        
        **🔍 Web Search Integration**
        - Latest information
        - Verified sources
        - Comprehensive answers
        """)
    
    st.markdown("---")
    
    # Statistics or info cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(label="🎓 Your Semester", value=user['study_year'])
    
    with col2:
        st.metric(label="📚 Available Semesters", value="4")
    
    with col3:
        st.metric(label="🤖 AI Models", value="Active")
    
    with col4:
        st.metric(label="✅ Status", value="Online")
    

def main():
    """Main function for testing"""
    st.set_page_config(
        page_title="UniMate AI - Dashboard",
        layout="wide",
        page_icon="🎓"
    )
    
    # For testing purposes - simulate logged in user
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = True
    
    if "user" not in st.session_state:
        st.session_state.user = {
            'name': 'Test User',
            'email': 'test@example.com',
            'college': 'Test College',
            'study_year': '5th Semester',
            'department': 'Computer Science',
            'phone': '9876543210'
        }
    
    show_semester_selection()

if __name__ == "__main__":
    main()