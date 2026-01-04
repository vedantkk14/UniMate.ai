"""
Login and Signup Page for UniMate AI
Entry point for the application
Save this file as: login.py
"""

import streamlit as st
import sys
import os

# Add current directory to path to import auth module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from auth import AuthSystem
except ImportError:
    st.error("❌ Error: auth.py not found. Please make sure auth.py is in the same directory as login.py")
    st.stop()

def show_login_page():

    # Hide default Streamlit navigation
    st.markdown("""
        <style>
        [data-testid="stSidebarNav"] {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Initialize auth system
    auth = AuthSystem()
    
    # Custom CSS for styling
    st.markdown("""
        <style>
        .main-header {
            text-align: center;
            color: #1f77b4;
            font-size: 3em;
            font-weight: bold;
            margin-bottom: 10px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .sub-header {
            text-align: center;
            color: #666;
            font-size: 1.3em;
            margin-bottom: 40px;
            font-style: italic;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
            background-color: #f0f2f6;
            border-radius: 10px;
            padding: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding: 10px 20px;
            background-color: white;
            border-radius: 8px;
            font-weight: 600;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4 !important;
            color: white !important;
        }
        div[data-testid="stForm"] {
            background-color: #f9f9f9;
            padding: 30px;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stButton > button {
            border-radius: 8px;
            height: 50px;
            font-weight: 600;
            font-size: 16px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🎓 UniMate AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Your Smart Learning Companion</p>', unsafe_allow_html=True)
    
    # Create tabs for Login and Sign Up
    tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
    
    # ================== LOGIN TAB ==================
    with tab1:
        st.markdown("### Welcome Back! 👋")
        st.markdown("Enter your credentials to continue your learning journey")
        
        with st.form("login_form"):
            email = st.text_input(
                "📧 Email Address",
                placeholder="your.email@example.com",
                key="login_email"
            )
            password = st.text_input(
                "🔒 Password",
                type="password",
                placeholder="Enter your password",
                key="login_password"
            )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit = st.form_submit_button("🚀 Login", use_container_width=True, type="primary")
            
            if submit:
                if not email or not password:
                    st.error("⚠️ Please fill in all fields")
                elif not auth.validate_email(email):
                    st.error("⚠️ Please enter a valid email address")
                else:
                    with st.spinner("🔍 Verifying credentials..."):
                        success, user_data = auth.login_user(email, password)
                        
                        if success:
                            # Store user data in session state
                            st.session_state.authenticated = True
                            st.session_state.user = user_data
                            st.success(f"✅ Welcome back, {user_data['name']}!")
                            st.balloons()
                            
                            # Small delay for effect
                            import time
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid email or password. Please try again or sign up if you're new.")
        
        st.markdown("---")
        st.info("💡 **New to UniMate AI?** Switch to the Sign Up tab to create your account!")
    
    # ================== SIGN UP TAB ==================
    with tab2:
        st.markdown("### Create Your Account 🌟")
        st.markdown("Join thousands of students already learning smarter!")
        
        with st.form("signup_form"):
            st.markdown("#### 📋 Personal Information")
            
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input(
                    "👤 Full Name *",
                    placeholder="John Doe",
                    key="signup_name"
                )
                email = st.text_input(
                    "📧 Email Address *",
                    placeholder="your.email@example.com",
                    key="signup_email"
                )
                password = st.text_input(
                    "🔒 Password *",
                    type="password",
                    placeholder="Minimum 6 characters",
                    key="signup_password",
                    help="Choose a strong password with at least 6 characters"
                )
                
            with col2:
                college = st.text_input(
                    "🏫 College/University *",
                    placeholder="ABC Engineering College",
                    key="signup_college"
                )
                study_year = st.selectbox(
                    "📚 Current Semester *",
                    ["Select your semester", "1st Semester", "2nd Semester", "3rd Semester", 
                     "4th Semester", "5th Semester", "6th Semester", 
                     "7th Semester", "8th Semester"],
                    key="signup_year"
                )
                department = st.text_input(
                    "🎓 Department/Branch",
                    placeholder="Computer Science (Optional)",
                    key="signup_dept"
                )
            
            st.markdown("#### 📞 Contact Information")
            phone = st.text_input(
                "📱 Phone Number",
                placeholder="10-digit number (Optional)",
                key="signup_phone",
                max_chars=10
            )
            
            st.markdown("---")
            st.markdown("_* Required fields_")
            
            # Terms checkbox
            terms = st.checkbox("I agree to the Terms of Service and Privacy Policy")
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                submit = st.form_submit_button(
                    "🎉 Create Account",
                    use_container_width=True,
                    type="primary"
                )
            
            if submit:
                # Validation
                if not terms:
                    st.error("⚠️ Please agree to the Terms of Service to continue")
                elif not all([name, email, password, college]) or study_year == "Select your semester":
                    st.error("⚠️ Please fill in all required fields marked with *")
                elif not auth.validate_email(email):
                    st.error("⚠️ Please enter a valid email address (e.g., user@example.com)")
                elif len(password) < 6:
                    st.error("⚠️ Password must be at least 6 characters long for security")
                elif phone and not auth.validate_phone(phone):
                    st.error("⚠️ Phone number must be exactly 10 digits")
                elif auth.user_exists(email):
                    st.error("⚠️ This email is already registered. Please use the Login tab instead.")
                else:
                    with st.spinner("🎨 Creating your account..."):
                        # Register user
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
                            st.success("✅ " + message)
                            st.balloons()
                            st.info("👉 **Next Step:** Switch to the Login tab and enter your credentials!")
                            
                            # Show success message with user details
                            with st.expander("📋 Your Registration Details", expanded=True):
                                st.write(f"**Name:** {name}")
                                st.write(f"**Email:** {email}")
                                st.write(f"**College:** {college}")
                                st.write(f"**Semester:** {study_year}")
                                if department:
                                    st.write(f"**Department:** {department}")
                        else:
                            st.error("❌ " + message)
        
        st.markdown("---")
        st.info("💡 **Already have an account?** Switch to the Login tab to sign in!")

def show_semester_selection():
    """Show semester selection page after successful login"""
    try:
        import semester_selection
        semester_selection.show_semester_selection()
    except ImportError:
        st.error("❌ Error: semester_selection.py not found. Please make sure it's in the same directory.")
        st.info("Creating a temporary dashboard...")
        
        # Fallback simple dashboard
        user = st.session_state.user
        st.title(f"Welcome, {user['name']}! 👋")
        st.write(f"**Email:** {user['email']}")
        st.write(f"**College:** {user['college']}")
        st.write(f"**Semester:** {user['study_year']}")
        
        if st.button("Go to 5th Semester"):
            try:
                st.switch_page("pages/home.py")
            except:
                st.error("pages/home.py not found")
        
        if st.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.user = None
            st.rerun()

def main():
    """Main function to run the login page"""
    st.set_page_config(
        page_title="UniMate AI - Login",
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
        # User is logged in, show semester selection
        show_semester_selection()

if __name__ == "__main__":
    main()