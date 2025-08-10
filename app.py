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
# Load environment variables from a .env file at the start of the app
load_dotenv()

# --- Caching Initializers ---
@st.cache_resource
def load_langchain_pipeline():
    """
    Loads the LangChain/Pinecone pipeline components.
    Reads API keys directly from environment variables.
    """
    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not pinecone_api_key or not google_api_key:
        raise ValueError("PINECONE_API_KEY and GOOGLE_API_KEY must be set in your .env file for this pipeline.")

    # Set environment variables for the current run, required by LangChain components
    os.environ["PINECONE_API_KEY"] = pinecone_api_key
    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    logging.info("Initializing LangChain + Pinecone pipeline...")
    llm, vectorstore = init_langchain()
    chain = create_rag_chain_with_sources(llm, vectorstore)
    logging.info("LangChain + Pinecone pipeline loaded successfully.")
    return chain

@st.cache_resource
def load_local_hybrid_pipeline():
    """
    Loads the Local Hybrid Search pipeline.
    Reads the Google API key directly from environment variables.
    """
    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY must be set in your .env file for this pipeline.")

    os.environ["GOOGLE_API_KEY"] = google_api_key
    
    logging.info("Initializing Local Hybrid Search pipeline...")
    pipeline = SummarizationPipeline()
    logging.info("Local Hybrid Search pipeline loaded successfully.")
    return pipeline

# --- Main Application UI ---
st.title("📄 Document Search and Summarization Engine")
st.markdown("" \
"This interface allows you to query a knowledge base using two different backend retrieval systems." \
" Checkout the project at- https://github.com/kawaiides/RAG" \
". It is adviced to use your own API keys as mine is on the free tier and subject to rate limits." \
". Made with ❤️ by Shyam Sunder | f20190644g@alumni.bits-pilani.ac.in"
)
# --- Sidebar Configuration ---
st.sidebar.header("⚙️ Configuration")
st.sidebar.info("API keys are loaded securely from your `.env` file.")

pipeline_option = st.sidebar.radio(
    "Choose a search pipeline:",
    ("LangChain + Pinecone (Cloud)", "Local Hybrid Search (BM25 + FAISS)"),
    key="pipeline_choice"
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
            # Execute the selected pipeline
            if pipeline_option == "LangChain + Pinecone (Cloud)":
                st.info("Using the **LangChain + Pinecone** pipeline...")
                with st.spinner("Retrieving documents and generating summary..."):
                    rag_chain = load_langchain_pipeline()
                    rag_output = rag_chain.invoke({"question": query})
                    final_answer = format_final_output(rag_output)

            elif pipeline_option == "Local Hybrid Search (BM25 + FAISS)":
                st.info("Using the **Local Hybrid Search** pipeline...")
                with st.spinner("Performing hybrid search and generating summary..."):
                    local_pipeline = load_local_hybrid_pipeline()
                    final_answer = local_pipeline.summarize(query)
            
            if final_answer:
                st.markdown("### Answer")
                st.markdown(final_answer, unsafe_allow_html=True)

        except ValueError as ve:
            # Catch missing API key errors specifically
            st.error(f"Configuration Error: {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logging.error(f"Error processing query '{query}': {e}", exc_info=True)
    else:
        st.warning("Please enter a question.")