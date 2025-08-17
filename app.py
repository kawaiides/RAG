# app.py
import streamlit as st
import logging
import os
import time
from dotenv import load_dotenv

# Import the necessary components from your pipeline scripts
# LangChain/Pinecone Pipeline
from src.langchain_query_engine import initialize_components as init_langchain, create_rag_chain_with_sources
# Local Hybrid Search Pipeline
from src.summarizer import SummarizationPipeline
# Uploaded File Pipeline
from src.uploaded_file_pipeline import create_pipeline_for_pdf

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

# --- Session State Initialization ---
if 'answer' not in st.session_state:
    st.session_state.answer = ""
if 'results' not in st.session_state:
    st.session_state.results = []
if 'page_number' not in st.session_state:
    st.session_state.page_number = 0
if 'history' not in st.session_state:
    st.session_state.history = []
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}

# --- Helper Functions ---
def normalize_results(results, pipeline_type, uploaded_filename=None):
    """Converts results from different pipelines into a standard format."""
    if not results:
        return []
    
    if pipeline_type == "LangChain + Pinecone (Cloud)":
        return [
            {
                'text': doc.page_content,
                'source_title': doc.metadata.get('source_title', 'Unknown Source'),
                'source_url': doc.metadata.get('source_url', '#'),
                'score': doc.metadata.get('score', 'N/A')
            } for doc in results
        ]
    
    elif pipeline_type == "Local Hybrid Search (BM25 + FAISS)":
        return results

    elif pipeline_type == "Search Uploaded PDF":
        return [
            {
                'text': doc.page_content,
                'source_title': f"Page {doc.metadata.get('page', 'N/A')} of {uploaded_filename}",
                'source_url': '#',
                'score': doc.metadata.get('score', 'N/A')
            } for doc in results
        ]
        
    return []

def display_metrics():
    """Displays the performance metrics at the top of the sidebar."""
    st.sidebar.header("📊 Performance Metrics")
    metrics = st.session_state.get('metrics', {})
    if not metrics:
        st.sidebar.write("No metrics yet.")
        return

    for key, value in metrics.items():
        st.sidebar.metric(label=key, value=value)

def display_paginated_results():
    """Displays the retrieved documents with pagination."""
    st.markdown("### Retrieved Documents")

    results = st.session_state.get('results', [])
    total_pages = len(results)

    if total_pages == 0:
        st.write("No documents found.")
        return

    # Reset page number if it's out of bounds
    if st.session_state.page_number >= total_pages:
        st.session_state.page_number = 0

    # --- Pagination Controls ---
    col1, col2, col3 = st.columns([2, 1, 2])
    
    with col1:
        if st.button("⬅️ Previous", use_container_width=True):
            if st.session_state.page_number > 0:
                st.session_state.page_number -= 1
                st.rerun()

    with col2:
        st.write(f"Page {st.session_state.page_number + 1} of {total_pages}")

    with col3:
        if st.button("Next ➡️", use_container_width=True):
            if st.session_state.page_number < total_pages - 1:
                st.session_state.page_number += 1
                st.rerun()

    # --- Display Document ---
    result = results[st.session_state.page_number]
    title = result.get('source_title', 'Unknown Source')
    url = result.get('source_url', '#')
    text = result.get('text', 'No content available.')

    st.markdown(f"#### [{title}]({url})")
    st.markdown(f"<div style='height: 300px; overflow-y: scroll; border: 1px solid #ccc; padding: 10px; border-radius: 5px;'>{text}</div>", unsafe_allow_html=True)


# --- Main Application UI ---
st.title("📄 Document Search and Summarization Engine")
st.markdown("" \
"This interface allows you to query a knowledge base using two different backend retrieval systems." \
" Checkout the project at- https://github.com/kawaiides/RAG ." \
" It is adviced to use your own API keys as mine is on the free tier and subject to rate limits." \
" Made with ❤️ by Shyam Sunder | f20190644g@alumni.bits-pilani.ac.in"
)
# --- Sidebar Configuration ---
display_metrics()

pipeline_option = st.sidebar.radio(
    "Choose a search pipeline:",
    ("LangChain + Pinecone (Cloud)", "Local Hybrid Search (BM25 + FAISS)", "Search Uploaded PDF"),
    key="pipeline_choice"
)

uploaded_file = None
if pipeline_option == "Search Uploaded PDF":
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

summary_length = st.sidebar.selectbox(
    "Select summary length:",
    ("Short", "Medium", "Long"),
    index=1,  # Default to Medium
    key="summary_length"
)

st.sidebar.header("📜 Conversation History")
if not st.session_state.history:
    st.sidebar.write("No history yet.")
else:
    for i, entry in enumerate(st.session_state.history):
        with st.sidebar.expander(f"Q: {entry['question']}"):
            st.markdown(entry['answer'])

st.sidebar.header("⚙️ Configuration")
st.sidebar.info("API keys are loaded securely from your `.env` file.")

# --- Main Query Area ---
query = st.text_input(
    "Enter your question:",
    placeholder="e.g., What is the attention mechanism and why is it important for transformers?",
    key="query_input"
)

if st.button("Get Answer", type="primary"):
    if query:
        # Reset state for new query
        st.session_state.answer = ""
        st.session_state.results = []
        st.session_state.page_number = 0
        st.session_state.metrics = {}
        
        start_time = time.time()
        try:
            if pipeline_option == "Search Uploaded PDF":
                if uploaded_file is not None:
                    st.info("Using the **Uploaded PDF** pipeline...")
                    with st.spinner("Processing PDF and generating answer..."):
                        google_api_key = os.getenv("GOOGLE_API_KEY")
                        if not google_api_key:
                            st.error("GOOGLE_API_KEY must be set in your .env file.")
                        else:
                            qa_chain = create_pipeline_for_pdf(uploaded_file, google_api_key)
                            output = qa_chain.invoke({"query": query})
                            st.session_state.answer = output.get('answer', "No answer found.")
                            st.session_state.results = normalize_results(output.get('docs', []), pipeline_option, uploaded_file.name)
                else:
                    st.warning("Please upload a PDF file to use this pipeline.")

            elif pipeline_option == "LangChain + Pinecone (Cloud)":
                st.info("Using the **LangChain + Pinecone** pipeline...")
                with st.spinner("Retrieving documents and generating summary..."):
                    rag_chain = load_langchain_pipeline()
                    rag_output = rag_chain.invoke({"question": query, "summary_length": summary_length})
                    
                    st.session_state.answer = rag_output.get('answer', "No answer generated.")
                    st.session_state.results = normalize_results(rag_output.get('docs', []), pipeline_option)

            elif pipeline_option == "Local Hybrid Search (BM25 + FAISS)":
                st.info("Using the **Local Hybrid Search** pipeline...")
                with st.spinner("Performing hybrid search and generating summary..."):
                    local_pipeline = load_local_hybrid_pipeline()
                    summary, search_results = local_pipeline.summarize(query, summary_length=summary_length)
                    
                    st.session_state.answer = summary
                    st.session_state.results = normalize_results(search_results, pipeline_option)

        except ValueError as ve:
            st.error(f"Configuration Error: {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logging.error(f"Error processing query '{query}': {e}", exc_info=True)
        
        end_time = time.time()
        st.session_state.metrics['Execution Time (s)'] = f"{end_time - start_time:.2f}"

        if st.session_state.answer:
            st.session_state.history.insert(0, {"question": query, "answer": st.session_state.answer})

    else:
        st.warning("Please enter a question.")

# --- Display Results ---
if st.session_state.answer:
    st.markdown("### Answer")
    st.markdown(st.session_state.answer, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("### Sources")
    
    results = st.session_state.get('results', [])
    if not results:
        st.write("No sources found.")
    else:
        for res in results:
            title = res.get('source_title', 'Unknown Source')
            url = res.get('source_url', '#')
            # This line correctly prioritizes 'rerank_score' for the hybrid search
            # and falls back to 'score' for other pipelines.
            score = res.get('rerank_score') or res.get('score')
            
            if score is not None and isinstance(score, (int, float)):
                st.markdown(f"* [{title}]({url}) (Score: {score:.4f})")
            else:
                st.markdown(f"* [{title}]({url})")

    if results:
        display_paginated_results()