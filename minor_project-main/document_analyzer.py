import os
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Configure Gemini API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to extract text from PDFs
def extract_text_from_pdfs(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        try:
            pdf_reader = PdfReader(pdf)
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            st.error(f"Error reading file '{pdf.name}': {str(e)}. Please check the file and try again.")
    return text


# Function to split text into chunks
def split_text_into_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)


# Function to create and save a FAISS vector store
def create_and_save_vector_store(text_chunks):
    if not text_chunks:
        raise ValueError("No valid text chunks found. Please ensure the files contain readable text.")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    vector_store.save_local("faiss_index")


# Function to define a conversation chain for QA
def create_conversation_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. If the answer is not in the context, just say, "Answer is not available in the context."
    Context:\n{context}\n
    Question:\n{question}\n
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)


# Function to handle user questions and provide answers
def handle_user_question(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = vector_store.similarity_search(user_question)
    chain = create_conversation_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    return response["output_text"]


# Main function to set up Streamlit UI for document analysis
def main():
    st.sidebar.title("Document-Analyzer")
    pdf_docs = st.sidebar.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])

    if st.sidebar.button("Submit & Process"):
        with st.spinner("Processing..."):
            raw_text = ""

            # Extract text from PDFs
            if pdf_docs:
                raw_text += extract_text_from_pdfs(pdf_docs)

            if not raw_text.strip():
                st.sidebar.error("No valid text could be extracted from the uploaded files.")
                return

            text_chunks = split_text_into_chunks(raw_text)
            try:
                create_and_save_vector_store(text_chunks)
                st.sidebar.success("Processing complete. You can now ask questions!")
            except ValueError as e:
                st.sidebar.error(str(e))

    with st.expander("Ask a Question:"):
        user_question = st.text_input("Enter your question here:")
        if user_question:
            with st.spinner("Fetching response..."):
                response = handle_user_question(user_question)
                st.write("Reply:", response)


# Run the app
if __name__ == "__main__":
    main()
