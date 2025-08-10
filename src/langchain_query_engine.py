import os
from dotenv import load_dotenv
from operator import itemgetter

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser

import logging

# --- 1. Initialization & Configuration ---
def initialize_components():
    """Load environment variables and initialize LangChain components."""
    load_dotenv()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in .env file")

    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
    logging.info("Google Gemini LLM initialized.")

    model_name = 'all-MiniLM-L6-v2'
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    logging.info(f"HuggingFace embedding model '{model_name}' loaded.")

    index_name = "ai-wikipedia"
    vectorstore = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    logging.info(f"Connected to Pinecone index '{index_name}'.")

    return llm, vectorstore

# --- 2. Building the RAG Chain with Citations ---
def create_rag_chain_with_sources(llm, vectorstore):
    """Create a RAG chain that includes source documents in the final output."""
    
    retriever = vectorstore.as_retriever(search_kwargs={'k': 5})
    logging.info("Retriever created from Pinecone vector store.")

    # The prompt remains the same, focused on answering from context
    prompt_template = """
You are an expert assistant. Answer the user's question based only on the following context.
If the context doesn't contain the answer, state that you cannot find the answer in the provided documents.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def format_docs(docs):
        """Helper function to format document content for the context."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    # This is the key change: a Runnable that retrieves documents and passes them through
    retrieval_chain = RunnablePassthrough.assign(
        docs=itemgetter("question") | retriever
    )

    # Now, build the main chain that uses the retrieved docs
    rag_chain_with_sources = (
        retrieval_chain
        | RunnablePassthrough.assign(
            context=(lambda x: format_docs(x["docs"]))
        )
        | {
            "answer": prompt | llm | StrOutputParser(),
            "docs": itemgetter("docs") # Pass the retrieved docs through to the end
        }
    )
    logging.info("RAG chain with source passthrough constructed successfully.")
    
    return rag_chain_with_sources

def format_final_output(rag_output: dict) -> str:
    """Formats the RAG output to include the answer and source links."""
    answer = rag_output['answer']
    docs = rag_output['docs']

    # Collect unique sources to avoid duplicates
    sources = {} # Use a dict to store title -> url
    for doc in docs:
        # Assumes metadata has 'source_title' and 'source_url' keys
        title = doc.metadata.get('source_title', 'Unknown Source')
        url = doc.metadata.get('source_url', '#')
        sources[title] = url

    if not sources:
        return answer

    # Format the source links
    formatted_sources = []
    for title, url in sources.items():
        formatted_sources.append(f"* [{title}]({url})")
    
    final_output = (
        f"{answer}\n\n"
        "---\n"
        "**Sources:**\n"
        + "\n".join(formatted_sources)
    )
    return final_output

# --- 3. Main Execution ---
if __name__ == "__main__":
    try:
        llm, vectorstore = initialize_components()
        rag_chain = create_rag_chain_with_sources(llm, vectorstore)

        question = "What is a transformer model in the context of artificial intelligence?"
        print("="*80)
        print(f"Question: {question}")
        
        # Invoke the chain to get the dictionary output
        rag_output = rag_chain.invoke({"question": question})
        
        # Format the final answer with sources
        final_answer = format_final_output(rag_output)
        
        print(f"\nFinal Answer:\n{final_answer}")
        print("="*80)

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)