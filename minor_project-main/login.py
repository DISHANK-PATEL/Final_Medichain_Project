import streamlit as st
from pymongo import MongoClient

# Connect to MongoDB
client = MongoClient('mongodb+srv://vashutyagi20:XQTUXO9RasveyVSG@cluster0.5y6ul.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0')
db = client['paw_some_ai']
collection = db['users']


def signup():
    st.title("Signup for Medichain")

    # Input fields
    username = st.text_input("Create Username")
    phone_number = st.text_input("Phone Number")
    aadhar_number = st.text_input("Aadhar Number")
    password = st.text_input("Create Password", type="password")

    if st.button("Signup"):
        # Validation checks
        if collection.find_one({"username": username}):
            st.error("Username already exists. Please choose a different one.")
        elif not username or not phone_number or not aadhar_number or not password:
            st.error("Please fill in all fields.")
        else:
            # Save the new user in MongoDB
            user_data = {
                "username": username,
                "phone_number": phone_number,
                "aadhar_number": aadhar_number,
                "password": password,
                "animals": []  # Initialize an empty animals array
            }
            collection.insert_one(user_data)
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Signup successful! You are now logged in.")


def login():
    st.title("Login to Medichain")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        user = collection.find_one({"username": username})
        if user and user["password"] == password:
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            st.success("Login successful!")
        else:
            st.error("Invalid username or password.")


def logout():
    st.session_state['logged_in'] = False
    st.session_state['username'] = None
    st.info("You have been logged out.")