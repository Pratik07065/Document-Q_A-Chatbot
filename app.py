import streamlit as st
import os
from pypdf import PdfReader
from docx import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Cached local embedding model loader for fast performance
@st.cache_resource
def get_hf_embeddings(model_name="all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

# Page setup
st.set_page_config(page_title="Document RAG Chatbot", page_icon="📄", layout="wide")
st.title("Document Q&A (PDF & Word)")

# Sidebar for API key and File Upload
with st.sidebar:
    st.header("Configuration")
    api_key = st.text_input("Enter Gemini API Key:", type="password")
    
    embedding_provider = st.radio(
        "Embedding Method:",
        ["Local Embeddings", "Gemini Cloud API"],
        index=0,
        help="Local embeddings (like Sentence-Transformers / all-MiniLM) run directly on your computer without hitting API quotas or model not found errors."
    )
    
    if "Local" in embedding_provider:
        hf_model = st.selectbox(
            "Local Model:",
            ["all-MiniLM-L6-v2", "BAAI/bge-small-en-v1.5"],
            index=0,
            help="'all-MiniLM-L6-v2' is a lightweight, high-performance transformer embedding model (the modern neural evolution of Word2Vec)."
        )
    else:
        gemini_model = st.selectbox(
            "Gemini Embedding Model:",
            ["models/text-embedding-004","gemini-embedding-001", "gemini-embedding-2"],
            index=0,
            help="'models/text-embedding-004' is the only official production embedding model for Gemini API."
        )
    
    uploaded_files = st.file_uploader(
        "Upload PDF or DOCX files", 
        type=["pdf", "docx"], 
        accept_multiple_files=True
    )
    process_button = st.button("Process Documents")

# Helper: Extract text from PDF & DOCX
def extract_text(files):
    text = ""
    for file in files:
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    text += extracted + "\n"
        elif file.name.endswith(".docx"):
            doc = Document(file)
            for para in doc.paragraphs:
                if para.text:
                    text += para.text + "\n"
    return text

# Helper: Format retrieved document chunks into a single context string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Process and index documents
if process_button:
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
    elif not uploaded_files:
        st.error("Please upload at least one document.")
    else:
        with st.spinner("Extracting text and building vector index..."):
            try:
                raw_text = extract_text(uploaded_files)
                
                if not raw_text.strip():
                    st.error("No readable text found in the uploaded documents.")
                else:
                    # 1. Chunking
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000,
                        chunk_overlap=200,
                        separators=["\n\n", "\n", ".", " ", ""]
                    )
                    chunks = text_splitter.split_text(raw_text)

                    # 2. Embeddings & FAISS Vector Store
                    if "Local" in embedding_provider:
                        embeddings = get_hf_embeddings(hf_model)
                    else:
                        embeddings = GoogleGenerativeAIEmbeddings(
                            model=gemini_model, 
                            google_api_key=api_key
                        )
                    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
                    
                    # 3. Store in session state
                    st.session_state.retriever = vector_store.as_retriever(
                        search_type="similarity", 
                        search_kwargs={"k": 3}
                    )
                    st.session_state.processed = True
                    st.success(f"Indexed {len(chunks)} chunks successfully!")
            except Exception as e:
                st.error(f"Error processing documents: {e}")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display conversation history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User question handling
user_query = st.chat_input("Ask a question about your documents...")

if user_query:
    if not api_key:
        st.error("Provide an API key to continue.")
    elif "retriever" not in st.session_state:
        st.warning("Please upload and process documents first.")
    else:
        # Display user question
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # 4. Strict RAG prompt (avoids hallucination / answers only from context)
        system_prompt = (
            "You are a strict assistant for question-answering tasks. "
            "Use ONLY the following pieces of retrieved context to answer the question. "
            "If the answer cannot be found in the context, state: "
            "'The provided document does not contain information to answer this question.' "
            "Do not make assumptions or use external knowledge.\n\n"
            "Context:\n{context}"
        )
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{question}"),
        ])

        # 5. Gemini LLM setup
        llm = ChatGoogleGenerativeAI(
            model="models/gemini-3.6-flash", 
            google_api_key=api_key, 
            temperature=0.0
        )

        # 6. Modern LCEL RAG Chain
        rag_chain = (
            {"context": st.session_state.retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        with st.chat_message("assistant"):
            with st.spinner("Generating grounded answer..."):
                try:
                    answer = rag_chain.invoke(user_query)
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"Error generating answer: {e}")