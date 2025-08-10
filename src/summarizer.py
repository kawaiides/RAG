import os
import logging
from dotenv import load_dotenv

# Import the HybridSearcher class from your local search implementation
from src.local_hybrid_search import HybridSearcher, INDEX_DIR

# We'll use LangChain's wrapper for the LLM
from langchain_google_genai import ChatGoogleGenerativeAI

class SummarizationPipeline:
    """
    Orchestrates the full RAG pipeline: search, context assembly, and summarization.
    """

    def __init__(self):
        logging.info("Initializing the summarization pipeline...")
        # 1. Initialize the searcher to find relevant documents
        self.searcher = HybridSearcher(INDEX_DIR)
        
        # 2. Initialize the LLM for generating the summary
        load_dotenv()
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.1)
        logging.info("Summarization pipeline initialized successfully.")

    def summarize(self, query: str) -> str:
        """
        Executes the full search-and-summarize pipeline for a given query.
        
        Args:
            query (str): The user's question.

        Returns:
            str: A formatted string containing the summary and source links.
        """
        logging.info(f"Executing pipeline for query: '{query}'")

        # Step 1: Search for relevant documents using our hybrid searcher
        search_results = self.searcher.search(query, top_k=5)

        if not search_results:
            return "Sorry, I could not find any relevant documents to answer your question."

        # Step 2: Assemble the context and collect unique sources
        context = "\n\n---\n\n".join([doc['text'] for doc in search_results])
        
        # Collect unique sources to avoid duplicates in the citation list
        sources = {} # Use a dict to store title -> url for automatic deduplication
        for doc in search_results:
            sources[doc['source_title']] = doc.get('source_url', '#') # Use '#' if URL is missing
        
        # Step 3: Build the prompt for the LLM
        prompt_template = f"""
        You are an expert summarization assistant. Your task is to provide a concise and factual summary that directly answers the user's question, based ONLY on the context provided.
        Do not include any information that is not mentioned in the context.

        CONTEXT:
        ---
        {context}
        ---

        QUESTION: {query}

        SUMMARY:
        """

        # Step 4: Generate the summary with the LLM
        logging.info("Generating summary from context...")
        summary = self.llm.invoke(prompt_template).content
        
        # Step 5: Format the final output with the summary and source links 🔗
        formatted_sources = []
        for title, url in sources.items():
            formatted_sources.append(f"* [{title}]({url})")
        
        final_output = (
            f"{summary}\n\n"
            "---\n"
            "**Sources:**\n"
            + "\n".join(formatted_sources)
        )
        
        return final_output

# --- Example Usage ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Instantiate the full pipeline
    pipeline = SummarizationPipeline()

    # Define a query
    user_query = "What is the attention mechanism and why is it important for transformers?"

    # Run the pipeline and get the final, formatted answer
    final_answer = pipeline.summarize(user_query)

    # Print the result
    print("\n" + "="*80)
    print(f"QUESTION: {user_query}\n")
    print("ANSWER:")
    print(final_answer)
    print("="*80)