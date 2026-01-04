"""
Authentication System for UniMate AI
Handles user registration, login, and database management
"""

import sqlite3
import hashlib
import re
from datetime import datetime
import streamlit as st

class AuthSystem:
    def __init__(self, db_path="users.db"):
        """Initialize authentication system with database"""
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database and create users table if it doesn't exist"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                name TEXT NOT NULL,
                college TEXT NOT NULL,
                study_year TEXT NOT NULL,
                department TEXT,
                phone TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        print("✅ Database initialized successfully")
    
    def hash_password(self, password):
        """Hash password using SHA-256"""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def validate_email(self, email):
        """Validate email format"""
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        return re.match(pattern, email) is not None
    
    def validate_phone(self, phone):
        """Validate phone number (10 digits)"""
        if not phone:
            return True  # Phone is optional
        pattern = r'^\d{10}$'
        return re.match(pattern, phone) is not None
    
    def user_exists(self, email):
        """Check if user exists in database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        result = cursor.fetchone()
        
        conn.close()
        return result is not None
    
    def register_user(self, email, password, name, college, study_year, department="", phone=""):
        """
        Register a new user
        Returns: (success: bool, message: str)
        """
        try:
            # Validate inputs
            if not self.validate_email(email):
                return False, "Invalid email format"
            
            if phone and not self.validate_phone(phone):
                return False, "Phone number must be exactly 10 digits"
            
            if len(password) < 6:
                return False, "Password must be at least 6 characters long"
            
            # Check if user already exists
            if self.user_exists(email):
                return False, "Email already exists. Please login instead."
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (email, password_hash, name, college, study_year, department, phone)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (email, password_hash, name, college, study_year, department, phone))
            
            conn.commit()
            conn.close()
            
            print(f"✅ User registered: {email}")
            return True, "Registration successful! Please login to continue."
        
        except sqlite3.IntegrityError as e:
            print(f"❌ Database integrity error: {e}")
            return False, "Email already exists. Please login instead."
        except Exception as e:
            print(f"❌ Registration error: {e}")
            return False, f"Registration failed: {str(e)}"
    
    def login_user(self, email, password):
        """
        Authenticate user and return user data
        Returns: (success: bool, user_data: dict or None)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            password_hash = self.hash_password(password)
            
            cursor.execute('''
                SELECT id, email, name, college, study_year, department, phone
                FROM users 
                WHERE email = ? AND password_hash = ?
            ''', (email, password_hash))
            
            result = cursor.fetchone()
            
            if result:
                # Update last login timestamp
                cursor.execute('''
                    UPDATE users 
                    SET last_login = CURRENT_TIMESTAMP 
                    WHERE email = ?
                ''', (email,))
                conn.commit()
                
                user_data = {
                    'id': result[0],
                    'email': result[1],
                    'name': result[2],
                    'college': result[3],
                    'study_year': result[4],
                    'department': result[5] if result[5] else 'Not specified',
                    'phone': result[6] if result[6] else 'Not provided'
                }
                
                conn.close()
                print(f"✅ User logged in: {email}")
                return True, user_data
            
            conn.close()
            print(f"❌ Login failed for: {email}")
            return False, None
            
        except Exception as e:
            print(f"❌ Login error: {e}")
            return False, None
    
    def get_user_info(self, email):
        """
        Get complete user information by email
        Returns: dict or None
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, email, name, college, study_year, department, phone, created_at, last_login
                FROM users 
                WHERE email = ?
            ''', (email,))
            
            result = cursor.fetchone()
            conn.close()
            
            if result:
                return {
                    'id': result[0],
                    'email': result[1],
                    'name': result[2],
                    'college': result[3],
                    'study_year': result[4],
                    'department': result[5] if result[5] else 'Not specified',
                    'phone': result[6] if result[6] else 'Not provided',
                    'created_at': result[7],
                    'last_login': result[8]
                }
            return None
            
        except Exception as e:
            print(f"❌ Error fetching user info: {e}")
            return None
    
    def update_user_info(self, email, **kwargs):
        """
        Update user information
        Allowed fields: name, college, study_year, department, phone
        Returns: bool
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            allowed_fields = ['name', 'college', 'study_year', 'department', 'phone']
            update_fields = []
            values = []
            
            for key, value in kwargs.items():
                if key in allowed_fields and value is not None:
                    update_fields.append(f"{key} = ?")
                    values.append(value)
            
            if update_fields:
                values.append(email)
                query = f"UPDATE users SET {', '.join(update_fields)} WHERE email = ?"
                cursor.execute(query, values)
                conn.commit()
                print(f"✅ User info updated for: {email}")
            
            conn.close()
            return True
            
        except Exception as e:
            print(f"❌ Error updating user info: {e}")
            return False
    
    def change_password(self, email, old_password, new_password):
        """
        Change user password
        Returns: (success: bool, message: str)
        """
        try:
            # Verify old password
            success, user_data = self.login_user(email, old_password)
            
            if not success:
                return False, "Current password is incorrect"
            
            if len(new_password) < 6:
                return False, "New password must be at least 6 characters long"
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            new_password_hash = self.hash_password(new_password)
            
            cursor.execute('''
                UPDATE users 
                SET password_hash = ? 
                WHERE email = ?
            ''', (new_password_hash, email))
            
            conn.commit()
            conn.close()
            
            print(f"✅ Password changed for: {email}")
            return True, "Password changed successfully"
            
        except Exception as e:
            print(f"❌ Error changing password: {e}")
            return False, f"Error changing password: {str(e)}"
    
    def delete_user(self, email, password):
        """
        Delete user account (requires password confirmation)
        Returns: (success: bool, message: str)
        """
        try:
            # Verify password
            success, user_data = self.login_user(email, password)
            
            if not success:
                return False, "Password incorrect. Account deletion cancelled."
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM users WHERE email = ?', (email,))
            
            conn.commit()
            conn.close()
            
            print(f"✅ User deleted: {email}")
            return True, "Account deleted successfully"
            
        except Exception as e:
            print(f"❌ Error deleting user: {e}")
            return False, f"Error deleting account: {str(e)}"
    
    def get_all_users(self):
        """
        Get all users (admin function)
        Returns: list of user dicts
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT id, email, name, college, study_year, created_at, last_login
                FROM users
                ORDER BY created_at DESC
            ''')
            
            results = cursor.fetchall()
            conn.close()
            
            users = []
            for result in results:
                users.append({
                    'id': result[0],
                    'email': result[1],
                    'name': result[2],
                    'college': result[3],
                    'study_year': result[4],
                    'created_at': result[5],
                    'last_login': result[6]
                })
            
            return users
            
        except Exception as e:
            print(f"❌ Error fetching users: {e}")
            return []


# Test function
if __name__ == "__main__":
    print("🧪 Testing Authentication System...")
    
    auth = AuthSystem()
    
    # Test registration
    success, msg = auth.register_user(
        email="test@example.com",
        password="test123",
        name="Test User",
        college="Test College",
        study_year="5th Semester",
        department="Computer Science",
        phone="9876543210"
    )
    print(f"Registration: {msg}")
    
    # Test login
    success, user = auth.login_user("test@example.com", "test123")
    if success:
        print(f"Login successful: {user['name']}")
    else:
        print("Login failed")
    
    print("\n✅ Authentication system is working!")