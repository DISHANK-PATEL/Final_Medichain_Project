import streamlit as st
from login import collection  # Assuming you have imported the shared MongoDB collection from db.py
import base64


def add_animal():
    st.title("Add Your Animal")

    # Check if the user is logged in
    if 'username' in st.session_state:
        username = st.session_state['username']

        # Input fields for the animal
        animal_name = st.text_input("Animal Name")
        animal_type = st.text_input("Type of Animal")
        animal_age = st.number_input("Animal Age (in years)", min_value=0, max_value=100, step=1)

        # Allow uploading multiple PDF files for important documents
        uploaded_reports = st.file_uploader("Upload Animal Documents (PDFs)", type="pdf", accept_multiple_files=True)

        if st.button("Add Animal"):
            user_data = collection.find_one({"username": username})
            if user_data:
                # Prepare important documents data
                reports = [
                    {"name": report.name, "pdf_data": base64.b64encode(report.read()).decode('utf-8')}
                    for report in uploaded_reports
                ]

                new_animal = {
                    "name": animal_name,
                    "type": animal_type,
                    "age": animal_age,
                    "important_documents": reports  # Renamed to important_documents
                }

                # Update user's animals array
                collection.update_one(
                    {"username": username},
                    {"$push": {"animals": new_animal}}
                )
                st.success(f"Animal '{animal_name}' added successfully!")
            else:
                st.error("User not found. Please log in again.")
    else:
        st.error("Please log in to add an animal.")