import streamlit as st
from langchain.llms import OpenAI
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Set up page configuration
st.set_page_config(page_title="📄 Ask the Doc App")
st.title("📄 Ask the Doc App")

# File uploader
uploaded_file = st.file_uploader("Upload an article", type='txt')

# Question input
query_text = st.text_input(
    "Enter your question:",
    placeholder="Please provide a short summary or specific question.",
    disabled=not uploaded_file
)

# Function to process and respond
def generate_response(uploaded_file, openai_api_key, query_text):
    # Load document
    if uploaded_file is not None:
        documents = [uploaded_file.read().decode()]

        # Split into chunks
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.create_documents(documents)

        # Generate embeddings
        embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

        # Create vector store
        db = FAISS.from_documents(texts, embeddings)

        # Create retriever and QA chain
        retriever = db.as_retriever()
        qa = RetrievalQA.from_chain_type(
            llm=OpenAI(openai_api_key=openai_api_key),
            chain_type="stuff",
            retriever=retriever
        )

        return qa.run(query_text)

# Input form
result = []
with st.form("myform", clear_on_submit=True):
    openai_api_key = st.text_input(
        "OpenAI API Key",
        type='password',
        disabled=not (uploaded_file and query_text)
    )

    submitted = st.form_submit_button(
        "Submit",
        disabled=not (uploaded_file and query_text)
    )

    if submitted and openai_api_key.startswith("sk-"):
        with st.spinner("Generating response..."):
            response = generate_response(uploaded_file, openai_api_key, query_text)
            result.append(response)
        del openai_api_key

# Display answer
if len(result):
    st.info(result[0])



