import streamlit as st
import logging
import os
from dotenv import load_dotenv

# Import the necessary components from your two pipeline scripts
# LangChain/Pinecone Pipeline
from src.langchain_query_engine import initialize_components as init_langchain, create_rag_chain_with_sources, format_final_output
# Local Hybrid Search Pipeline
from src.summarizer import SummarizationPipeline

# --- App Configuration ---
st.set_page_config(
    page_title="Document Search & Summarization",
    page_icon="📚",
    layout="wide"
)

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Load API Keys ---
# Load from .env file once at the start
load_dotenv()

# --- Caching Initializers ---
# The caching function now depends on the API keys. If a key changes, Streamlit
# will clear the cache and re-run this function with the new key.
@st.cache_resource
def load_langchain_pipeline(pinecone_api_key, google_api_key):
    """Loads the LangChain/Pinecone pipeline components."""
    # Set environment variables for the current run, required by LangChain components
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    logging.info("Initializing LangChain + Pinecone pipeline...")
    llm, vectorstore = init_langchain()
    chain = create_rag_chain_with_sources(llm, vectorstore)
    logging.info("LangChain + Pinecone pipeline loaded successfully.")
    return chain

@st.cache_resource
def load_local_hybrid_pipeline(google_api_key):
    """Loads the Local Hybrid Search pipeline."""
    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    logging.info("Initializing Local Hybrid Search pipeline...")
    pipeline = SummarizationPipeline()
    logging.info("Local Hybrid Search pipeline loaded successfully.")
    return pipeline

# --- Main Application UI ---
st.title("📄 Document Search and Summarization Engine")
st.markdown("This interface allows you to query a knowledge base using two different backend retrieval systems.")

# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")

pipeline_option = st.sidebar.radio(
    "Choose a search pipeline:",
    ("LangChain + Pinecone (Cloud)", "Local Hybrid Search (BM25 + FAISS)"),
    key="pipeline_choice"
)

# --- API Key Inputs based on selection ---
if pipeline_option == "LangChain + Pinecone (Cloud)":
    st.sidebar.subheader("Pinecone Credentials")
    pinecone_api_key = st.sidebar.text_input(
        "Pinecone API Key", 
        type="password", 
        value=os.getenv("PINECONE_API_KEY", "")
    )
    st.sidebar.subheader("Google Gemini Credentials")
    google_api_key_pinecone = st.sidebar.text_input(
        "Google API Key", 
        type="password", 
        key="google_api_pinecone",
        value=os.getenv("GOOGLE_API_KEY", "")
    )
else: # Local Hybrid Search
    st.sidebar.subheader("Google Gemini Credentials")
    google_api_key_local = st.sidebar.text_input(
        "Google API Key", 
        type="password", 
        key="google_api_local",
        value=os.getenv("GOOGLE_API_KEY", "")
    )

# --- Main Query Area ---
query = st.text_input(
    "Enter your question:",
    placeholder="e.g., What is the attention mechanism and why is it important for transformers?",
    key="query_input"
)

if st.button("Get Answer", type="primary"):
    if query:
        final_answer = ""
        try:
            # Execute the selected pipeline with the provided keys
            if pipeline_option == "LangChain + Pinecone (Cloud)":
                if not all([pinecone_api_key, google_api_key_pinecone]):
                    st.error("Please provide all required API keys for the Pinecone pipeline in the sidebar.")
                else:
                    st.info("Using the **LangChain + Pinecone** pipeline...")
                    with st.spinner("Retrieving documents and generating summary..."):
                        rag_chain = load_langchain_pipeline(pinecone_api_key, google_api_key_pinecone)
                        rag_output = rag_chain.invoke({"question": query})
                        final_answer = format_final_output(rag_output)

            elif pipeline_option == "Local Hybrid Search (BM25 + FAISS)":
                if not google_api_key_local:
                    st.error("Please provide the Google API key for the Local Hybrid pipeline in the sidebar.")
                else:
                    st.info("Using the **Local Hybrid Search** pipeline...")
                    with st.spinner("Performing hybrid search and generating summary..."):
                        local_pipeline = load_local_hybrid_pipeline(google_api_key_local)
                        final_answer = local_pipeline.summarize(query)
            
            if final_answer:
                st.markdown("### Answer")
                st.markdown(final_answer, unsafe_allow_html=True)

        except Exception as e:
            st.error(f"An error occurred: {e}")
            logging.error(f"Error processing query '{query}': {e}", exc_info=True)
    else:
        st.warning("Please enter a question.")