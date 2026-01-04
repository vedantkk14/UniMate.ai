import sqlite3
import hashlib

# 1. Create/Connect to Database
def create_connection():
    # check_same_thread=False is needed for Streamlit's multi-threading
    conn = sqlite3.connect('student_data.db', check_same_thread=False)
    return conn

# 2. Create the User Table (Run this once on app start)
def create_table():
    conn = create_connection()
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            email TEXT PRIMARY KEY,
            password TEXT NOT NULL,
            name TEXT,
            college TEXT,
            year TEXT
        )
    ''')
    conn.commit()
    conn.close()

# 3. Add a New User (Signup)
def add_user(email, password, name, college, year):
    conn = create_connection()
    c = conn.cursor()
    # Basic hashing for security
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    
    try:
        c.execute('INSERT INTO users (email, password, name, college, year) VALUES (?, ?, ?, ?, ?)', 
                  (email, hashed_pw, name, college, year))
        conn.commit()
        success = True
    except sqlite3.IntegrityError:
        # Returns False if email already exists
        success = False
        
    conn.close()
    return success

# 4. Verify User (Login)
def login_user(email, password):
    conn = create_connection()
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    
    c.execute('SELECT * FROM users WHERE email = ? AND password = ?', (email, hashed_pw))
    data = c.fetchone()
    conn.close()
    return data