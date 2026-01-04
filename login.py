"""
Login and Signup Page for UniMate AI
Entry point for the application
Save this file as: login.py
"""

import streamlit as st
import sys
import os
import time

# Add current directory to path to import auth module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from auth import AuthSystem
except ImportError:
    st.error("❌ Error: auth.py not found. Please make sure auth.py is in the same directory as login.py")
    st.stop()

def show_login_page():
    # Minimal CSS just to hide the default sidebar navigation
    st.markdown("""
        <style>
        [data-testid="stSidebarNav"] {display: none;}
        .block-container {padding-top: 3rem;}
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize auth system
    auth = AuthSystem()
    
    # --- HEADER ---
    st.title("🎓 UniMate AI")
    st.markdown("### Your Smart Learning Companion")
    st.caption("Log in or create an account to access your semester dashboard.")
    
    # Create tabs for Login and Sign Up
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Create Account"])
    
    # ================== LOGIN TAB ==================
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True) # Spacer
        
        with st.form("login_form"):
            email = st.text_input("Email Address", placeholder="student@example.com", key="login_email")
            password = st.text_input("Password", type="password", placeholder="••••••••", key="login_password")
            
            # Simple, clean button
            submit = st.form_submit_button("🚀 Log In", use_container_width=True, type="primary")
            
            if submit:
                if not email or not password:
                    st.warning("⚠️ Please fill in all fields")
                elif not auth.validate_email(email):
                    st.warning("⚠️ Please enter a valid email address")
                else:
                    with st.spinner("Verifying..."):
                        success, user_data = auth.login_user(email, password)
                        
                        if success:
                            st.session_state.authenticated = True
                            st.session_state.user = user_data
                            st.success(f"✅ Welcome back, {user_data['name']}!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid email or password.")
        
        st.markdown("---")
        st.caption("New here? Switch to the **Create Account** tab.")

    # ================== SIGN UP TAB ==================
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True) # Spacer
        
        with st.form("signup_form"):
            st.markdown("#### 📋 Personal Information")
            
            # Using columns for a cleaner, non-scrolling layout
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Full Name *", placeholder="John Doe", key="signup_name")
                email = st.text_input("Email Address *", placeholder="john@college.edu", key="signup_email")
                password = st.text_input("Password *", type="password", help="Min 6 characters", key="signup_password")
                
            with col2:
                college = st.text_input("College/University *", placeholder="ABC Institute", key="signup_college")
                study_year = st.selectbox("Current Semester *", 
                    ["Select semester...", "1st Semester", "2nd Semester", "3rd Semester", 
                     "4th Semester", "5th Semester", "6th Semester", 
                     "7th Semester", "8th Semester"], key="signup_year")
                department = st.text_input("Department", placeholder="CS/IT (Optional)", key="signup_dept")
            
            # Full width field for phone
            phone = st.text_input("Phone Number", placeholder="10-digit number (Optional)", max_chars=10, key="signup_phone")
            
            st.markdown("---")
            
            # Terms checkbox
            terms = st.checkbox("I agree to the Terms of Service & Privacy Policy")
            
            submit = st.form_submit_button("✨ Create Account", use_container_width=True, type="primary")
            
            if submit:
                # Validation Logic (Kept exactly as requested)
                if not terms:
                    st.warning("⚠️ Please agree to the Terms of Service")
                elif not all([name, email, password, college]) or study_year == "Select semester...":
                    st.error("⚠️ Please fill in all required fields (*)")
                elif not auth.validate_email(email):
                    st.error("⚠️ Invalid email format")
                elif len(password) < 6:
                    st.error("⚠️ Password must be at least 6 characters")
                elif phone and not auth.validate_phone(phone):
                    st.error("⚠️ Phone number must be 10 digits")
                elif auth.user_exists(email):
                    st.error("⚠️ Email already registered. Please Login.")
                else:
                    with st.spinner("Creating account..."):
                        success, message = auth.register_user(
                            email=email.strip().lower(),
                            password=password,
                            name=name.strip(),
                            college=college.strip(),
                            study_year=study_year,
                            department=department.strip(),
                            phone=phone.strip()
                        )
                        
                        if success:
                            st.success(f"✅ {message}")
                            st.balloons()
                            st.info("👉 Switch to the **Login** tab to sign in!")
                        else:
                            st.error(f"❌ {message}")

def show_semester_selection():
    """Show semester selection page after successful login"""
    # Logic unchanged
    try:
        import semester_selection
        semester_selection.show_semester_selection()
    except ImportError:
        st.error("❌ Error: semester_selection.py not found.")
        
        # Fallback simple dashboard (Visual update only)
        user = st.session_state.user
        st.title(f"👋 Hi, {user['name']}")
        
        with st.container(border=True):
            st.info(f"🎓 **{user['college']}** | {user['study_year']}")
            st.write(f"📧 {user['email']}")
        
        st.markdown("### Quick Access")
        if st.button("📚 Go to 5th Semester", use_container_width=True, type="primary"):
            try:
                st.switch_page("pages/home.py")
            except:
                st.error("pages/home.py not found")
        
        if st.button("🔒 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()

def main():
    """Main function to run the login page"""
    st.set_page_config(
        page_title="UniMate AI",
        layout="centered",
        page_icon="🎓",
        initial_sidebar_state="collapsed"
    )
    
    # Initialize session state
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    
    if "user" not in st.session_state:
        st.session_state.user = None
    
    # Check authentication status
    if not st.session_state.authenticated:
        show_login_page()
    else:
        show_semester_selection()

if __name__ == "__main__":
    main()