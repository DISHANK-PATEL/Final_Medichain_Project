import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import pyttsx3
import speech_recognition as sr
from pydub import AudioSegment
import io

# Function to convert speech to text
def speech_to_text(audio_file):
    recognizer = sr.Recognizer()
    audio = AudioSegment.from_wav(io.BytesIO(audio_file.read()))
    audio_path = "temp.wav"
    audio.export(audio_path, format="wav")

    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return "Sorry, I couldn't understand the audio."
        except sr.RequestError:
            return "Sorry, the speech recognition service is unavailable."

def chatbot():
    # Importing all the modules
    import streamlit as st
    from PyPDF2 import PdfReader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    import os
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    import google.generativeai as genai
    from langchain.vectorstores import FAISS
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.chains.question_answering import load_qa_chain
    from langchain.prompts import PromptTemplate
    from dotenv import load_dotenv
    import pyttsx3

    def speak_response(response_content):
        engine = pyttsx3.init()
        engine.say(response_content)
        engine.runAndWait()

    # Load environment variables
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=api_key)

    def get_pdf_text(pdf_docs):
        text = ""
        for pdf in pdf_docs:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text

    def get_text_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
        chunks = text_splitter.split_text(text)
        return chunks

    def get_vector_store(text_chunks):
        embedding_function = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embedding_function)
        vector_store.save_local("faiss_index")

    def get_conversational_chain():
        prompt_template = """
        Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
        provided context then go and find and provide the answer don't provide the wrong answer and your a expert in any animal related matter so make sure all your responses are within that.\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0.3)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
        return chain

    def user_input(user_question):
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]

    # Main function for the chatbot
    st.title("Pet Care ChatBot 🐾")
    st.subheader("Your AI-Powered Pet Care Assistant")
    st.markdown("""Welcome to the Pet Care ChatBot! Ask any question related to pet care, and our AI-powered assistant will provide you with detailed and accurate answers.""")
    voice_response = st.checkbox("Click for Voice Response")

    # Upload audio file for speech-to-text
    uploaded_audio = st.file_uploader("Upload a .wav file", type=["wav"])

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if uploaded_audio:
        # Convert audio to text
        user_question = speech_to_text(uploaded_audio)
        st.chat_message("user").markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})

        # Process user input
        response = user_input(user_question)
        with st.chat_message("assistant"):
            st.markdown(response)
            if voice_response:
                speak_response(response)

        st.session_state.messages.append({"role": "assistant", "content": response})

    # React to user input via text
    elif prompt := st.chat_input("Ask a question from the PDF files"):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = user_input(prompt)
        with st.chat_message("assistant"):
            st.markdown(response)
            if voice_response:
                speak_response(response)

        st.session_state.messages.append({"role": "assistant", "content": response})
