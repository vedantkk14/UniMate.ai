"""
Semester Selection Page for UniMate AI
Displays after successful login
"""

import streamlit as st

def show_semester_selection():
    """Show semester selection dashboard after login"""

    # Minimal CSS to hide default navigation and adjust spacing
    st.markdown("""
        <style>
        [data-testid="stSidebarNav"] {display: none;}
        .block-container {padding-top: 3rem;}
        </style>
    """, unsafe_allow_html=True)
    
    # Get user data from session
    user = st.session_state.user
    
    # --- SIDEBAR PROFILE ---
    with st.sidebar:
        st.title("👤 Profile")
        st.subheader(user['name'])
        st.write(f"**🏫 College:** {user['college']}")
        st.write(f"**📧 Email:** {user['email']}")
        st.write(f"**📚 Current:** {user['study_year']}")
        if user.get('department') and user['department'] != 'Not specified':
            st.write(f"**🎓 Dept:** {user['department']}")
        
        st.markdown("---")
        
        # Settings (Visual only for now)
        st.markdown("### ⚙️ Settings")
        if st.button("✏️ Edit Profile", use_container_width=True):
            st.toast("Profile editing coming soon!", icon="🚧")
            
        if st.button("🔒 Change Password", use_container_width=True):
            st.toast("Password change coming soon!", icon="🚧")

        st.markdown("---")
        
        # Logout Logic
        if st.button("🚪 Logout", use_container_width=True, type="primary"):
            st.session_state.authenticated = False
            st.session_state.user = None
            # Clear chat history on logout
            if 'messages' in st.session_state:
                st.session_state.messages = []
            if 'chat_history' in st.session_state:
                st.session_state.chat_history = []
            st.rerun()

    # --- MAIN DASHBOARD CONTENT ---
    
    # Welcome Header
    st.title(f"👋 Welcome, {user['name'].split()[0]}!")
    st.markdown("#### Ready to continue your learning journey?")
    st.markdown("---")
    
    # 1. SEMESTER SELECTION (Using Containers for "Cards")
    st.subheader("📚 Select Your Learning Path")
    
    col1, col2 = st.columns(2)
    
    # 5th Semester Card
    with col1:
        with st.container(border=True):
            st.markdown("### 📙 5th Semester")
            st.caption("Access AI, CN, DBMS, and TOC resources.")
            st.markdown("<br>", unsafe_allow_html=True) # Spacer
            
            if st.button("Enter 5th Sem", use_container_width=True, type="primary", key="sem_5"):
                st.session_state.selected_semester = "5th Semester"
                st.switch_page("pages/sem5_home.py")
    
    # 6th Semester Card
    with col2:
        with st.container(border=True):
            st.markdown("### 📕 6th Semester")
            st.caption("Explore ML, Web Dev, and Cloud Computing.")
            st.markdown("<br>", unsafe_allow_html=True) # Spacer
            
            if st.button("Enter 6th Sem", use_container_width=True, key="sem_6"):
                st.session_state.selected_semester = "6th Semester"
                st.switch_page("pages/sem6_home.py")

# Future Semesters (Visual Placeholders)
    st.markdown("#### Upcoming Modules")
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        if st.button("7th Sem", use_container_width=True):
            st.toast("Sem 7th is coming soon!", icon="🚧")
            
    with c2:
        if st.button("8th Sem", use_container_width=True):
            st.toast("Sem 8th is coming soon!", icon="🚧")

    st.markdown("---")
    
    # 2. QUICK ACTIONS (Grid Layout)
    st.subheader("🎯 Quick Actions")
    
    qa1, qa2, qa3 = st.columns(3)
    
    with qa1:
        with st.container(border=True):
            st.markdown("#### 💬 AI Chatbot")
            st.caption("Ask academic questions.")
            if st.button("Start Chat", key="chat_btn", use_container_width=True):
                st.session_state.selected_semester = user['study_year']
                try:
                    st.switch_page("pages/home.py")
                except:
                    st.toast("Chatbot is being updated...", icon="⚙️")

    with qa2:
        with st.container(border=True):
            st.markdown("#### 📝 Quiz")
            st.caption("Test your knowledge.")
            if st.button("Take Quiz", key="quiz_btn", use_container_width=True):
                st.toast("Quiz module coming soon!", icon="🚧")

    with qa3:
        with st.container(border=True):
            st.markdown("#### 📊 Progress")
            st.caption("Track your stats.")
            if st.button("View Stats", key="progress_btn", use_container_width=True):
                st.toast("Analytics dashboard coming soon!", icon="🚧")

    st.markdown("---")

    # 3. STATS OVERVIEW
    st.subheader("Your Status")
    stat1, stat2, stat3, stat4 = st.columns(4)
    stat1.metric("🎓 Current Sem", user['study_year'])
    stat2.metric("📚 Modules", "4 Available")
    stat3.metric("🔥 Streak", "3 Days")
    stat4.metric("🤖 AI Status", "Online")

def main():
    """Main function for testing independent execution"""
    st.set_page_config(
        page_title="UniMate AI - Dashboard",
        layout="centered",
        page_icon="🎓"
    )
    
    # Mock data for testing
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = True
    
    if "user" not in st.session_state:
        st.session_state.user = {
            'name': 'Test Student',
            'email': 'test@example.com',
            'college': 'Demo Institute of Tech',
            'study_year': '5th Semester',
            'department': 'Computer Science',
            'phone': '1234567890'
        }
    
    show_semester_selection()

if __name__ == "__main__":
    main()