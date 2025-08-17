# src/uploaded_file_pipeline.py

import os
import tempfile
import asyncio
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def _format_docs_with_scores(docs_with_scores):
    """
    Helper function to add the retrieval score to each document's metadata.
    FAISS returns L2 distance, so a lower score is better.
    """
    formatted_docs = []
    for doc, score in docs_with_scores:
        doc.metadata['score'] = score
        formatted_docs.append(doc)
    return formatted_docs

def create_pipeline_for_pdf(uploaded_file, google_api_key: str):
    """
    Creates a complete RAG pipeline for a PDF that returns sources and scores.
    """
    # Handle asyncio event loop
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Use a temporary file to handle the uploaded PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
        tmpfile.write(uploaded_file.getvalue())
        tmp_path = tmpfile.name

    try:
        logging.info(f"Processing PDF from temporary path: {tmp_path}")
        
        # 1. Load and split the document
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()
        
        if not documents:
            raise ValueError("Could not load any content from the PDF.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)
        logging.info(f"Split PDF into {len(docs)} chunks.")

        # 2. Create embeddings and vector store
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        logging.info("Creating in-memory FAISS vector store...")
        vector_store = FAISS.from_documents(docs, embeddings)
        logging.info("FAISS vector store created successfully.")

        # 3. Define the prompt and LLM
        prompt_template = """
        You are an assistant for question-answering tasks.
        Use only the following pieces of retrieved context to answer the question.
        If you don't know the answer, just say that you don't know. Be concise.

        Context:
        {context}

        Question:
        {input}

        Answer:
        """
        prompt = ChatPromptTemplate.from_template(prompt_template)
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
        question_answer_chain = create_stuff_documents_chain(llm, prompt)

        # 4. Create a function to retrieve docs AND format them with scores
        def retrieve_docs_with_scores(input_dict):
            """Retrieves documents from the vector store and formats them with their scores."""
            query = input_dict["query"]
            # Use similarity_search_with_score to get (document, score) tuples
            docs_with_scores = vector_store.similarity_search_with_score(query, k=5)
            # Use the helper function to add scores to metadata
            return _format_docs_with_scores(docs_with_scores)

        # 5. Assemble the final chain using LangChain Expression Language (LCEL)
        rag_chain = (
            {
                # The 'context' is now the list of documents with scores in their metadata
                "context": RunnableLambda(retrieve_docs_with_scores),
                "input": RunnablePassthrough()
            }
            # Create a new 'answer' key by passing the context and original query to the QA chain
            | RunnablePassthrough.assign(
                answer=(lambda x: {"context": x["context"], "input": x["input"]["query"]}) | question_answer_chain
            )
            # The final output is a dictionary with the answer and the score-annotated documents
            | (lambda x: {"answer": x["answer"], "docs": x["context"]})
        )
        
        logging.info("PDF processing pipeline with scores created and ready.")
        return rag_chain

    except Exception as e:
        logging.error(f"Failed to create PDF pipeline: {e}", exc_info=True)
        raise
    finally:
        # Clean up the temporary file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            logging.info(f"Removed temporary file: {tmp_path}")