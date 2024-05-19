import streamlit as st
import hashlib
import sqlite3

# Function to create a hashed password
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

# Function to check if password matches the hashed password
def check_hashes(password, hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False

# Function to create user table
def create_usertable():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')
    conn.commit()
    conn.close()

# Function to add user data to the user table
def add_userdata(username, password):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()
    conn.close()

# Function to log in user
def login_user(username, password):
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,check_hashes(password)))
    data = c.fetchall()
    conn.close()
    return data

def view_all_users():
    conn = sqlite3.connect('data.db')
    c = conn.cursor()
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data


# Create user interface for login and signup page
def login():
    st.title("Login Page")
    username = st.text_input("Username")
    password = st.text_input("Password", type='password')
    if st.button("Login"):
        result = login_user(username, password)
        if result:
            st.success("Logged In as {}".format(username))
            # Store login state in a session variable or a cookie
        else:
            st.warning("Incorrect Username/Password")

def signup():
    st.title("Signup Page")
    new_user = st.text_input("Username")
    new_password = st.text_input("Password", type='password')
    if st.button("Signup"):
        create_usertable()
        add_userdata(new_user, make_hashes(new_password))
        st.success("You have successfully created an Account")
        st.info("Go to Login Menu to login")
