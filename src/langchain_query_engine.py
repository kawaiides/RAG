import os
from dotenv import load_dotenv
from operator import itemgetter
from typing import List, Dict, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document

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

# --- 2. Building the RAG Chain with Citations & Scores ---
def create_rag_chain_with_sources(llm, vectorstore):
    """Create a RAG chain that includes source documents with relevance scores."""
    
    retriever = vectorstore.as_retriever(
        search_kwargs={'k': 5}
    )
    logging.info("Retriever created from Pinecone vector store.")

    prompt_template = """
You are an expert assistant. Answer the user's question based only on the following context.
Your answer should {instruction}.
If the context doesn't contain the answer, state that you cannot find the answer in the provided documents.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""
    prompt = ChatPromptTemplate.from_template(prompt_template)

    def get_length_instruction(summary_length: str) -> str:
        length_instructions = {
            "Short": "be very concise, in one or two sentences",
            "Medium": "be detailed, in a few paragraphs",
            "Long": "be comprehensive, explaining all relevant details"
        }
        return length_instructions.get(summary_length, "be detailed, in a few paragraphs")

    def format_docs(docs: List[Document]) -> str:
        """Helper function to format document content for the context."""
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    def retrieve_and_add_scores(input_dict: Dict[str, Any]) -> List[Document]:
        """
        Custom retrieval function to get docs and add scores to metadata.
        """
        question = input_dict["question"]
        docs_with_scores = vectorstore.similarity_search_with_score(
            question, 
            k=retriever.search_kwargs.get('k', 5)
        )
        
        for doc, score in docs_with_scores:
            doc.metadata['score'] = score
        return [doc for doc, score in docs_with_scores]

    rag_chain_with_sources = (
        RunnablePassthrough.assign(
            docs=RunnableLambda(retrieve_and_add_scores),
            instruction=itemgetter("summary_length") | RunnableLambda(get_length_instruction)
        )
        | RunnablePassthrough.assign(
            context=(lambda x: format_docs(x["docs"]))
        )
        | {
            "answer": prompt | llm | StrOutputParser(),
            "docs": itemgetter("docs")
        }
    )
    logging.info("RAG chain with source passthrough and scores constructed successfully.")
    
    return rag_chain_with_sources


def format_final_output(rag_output: dict) -> str:
    """Formats the RAG output to include the answer and source links with scores."""
    answer = rag_output['answer']
    docs = rag_output['docs']

    if not docs:
        return answer

    # Collect unique sources with their scores to avoid duplicates
    sources = []
    seen_urls = set()
    for doc in docs:
        url = doc.metadata.get('source_url', '#')
        if url not in seen_urls:
            title = doc.metadata.get('source_title', 'Unknown Source')
            # Extract the relevance score from metadata
            score = doc.metadata.get('score')
            sources.append({'title': title, 'url': url, 'score': score})
            seen_urls.add(url)
    
    if not sources:
        return answer

    # Format the source links to include the score
    formatted_sources = []
    for source in sources:
        score_str = f" (Score: {source['score']:.2f})" if source['score'] is not None else ""
        formatted_sources.append(f"* [{source['title']}]({source['url']}){score_str}")
    
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
        rag_output = rag_chain.invoke({"question": question, "summary_length": "Short"})
        
        # Format the final answer with sources and scores
        final_answer = format_final_output(rag_output)
        
        print(f"\nFinal Answer:\n{final_answer}")
        print("="*80)

    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
