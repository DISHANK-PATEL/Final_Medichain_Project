import os
from PIL import Image
from io import BytesIO
import requests
import streamlit as st
import google.generativeai as genai
from streamlit_option_menu import option_menu

# Hardcoded API Key (Replace with your actual key)
GOOGLE_API_KEY = "AIzaSyAGSIthDU7Nfi6cOd3-FXPZIfA54NKIoOU"

# Configure the API key for generative AI
if not GOOGLE_API_KEY:
    st.error("Google API Key is missing. Please check your configuration.")
else:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    genai.configure(api_key=GOOGLE_API_KEY)

# Function to interact with the Gemini 1.5 Flash model
def gemini_flash(prompt, image):
    try:
        gemini_pro_vision_model = genai.GenerativeModel("gemini-1.5-flash")
        response = gemini_pro_vision_model.generate_content([prompt, image])
        return response.text
    except Exception as e:
        return f"Error generating content: {e}"

# Main function to run Image Captioning
def run():
    st.title("Search Via Image 📸")

    upload_option = st.radio("Choose image input method:", ("Upload Image", "Paste Image URL"))

    # Handle image upload
    image = None
    if upload_option == "Upload Image":
        uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            image = Image.open(BytesIO(uploaded_image.read()))
    elif upload_option == "Paste Image URL":
        image_url = st.text_input("Paste the image URL here:")
        if image_url:
            try:
                response = requests.get(image_url)
                image = Image.open(BytesIO(response.content))
            except Exception as e:
                st.error(f"Error loading image from URL: {e}")

    # Input prompt for the image
    user_prompt = st.text_input("Enter a custom prompt for image narration:", "Explain the image uploaded")

    # Button to generate analysis
    if st.button("Generate Analysis"):
        if image:
            col1, col2 = st.columns(2)

            # Display the uploaded or fetched image
            with col1:
                resized_image = image.resize((800, 500))
                st.image(resized_image, caption="Uploaded Image")

            # Generate and display AI response
            caption = gemini_flash(user_prompt, image)
            with col2:
                st.info(caption)
        else:
            st.warning("Please upload an image or paste an image URL.")

    # Custom CSS for styling
    st.markdown("""
        <style>
        body {
            font-size: 20px; /* Increase font size */
        }
        .css-1d391kg, .css-1v3fvcr {
            font-size: 28px; /* Customize headers */
        }
        </style>
    """, unsafe_allow_html=True)
