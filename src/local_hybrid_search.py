import json
import pickle
from pathlib import Path
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import logging

# --- Configuration ---
INDEX_DIR = Path("data/local_hybrid_index")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
# Cross-Encoders are trained to predict similarity scores for (query, document) pairs
RERANKER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class HybridSearcher:
    """Encapsulates the logic for hybrid search and reranking."""

    def __init__(self, index_directory: Path):
        logging.info("Loading models and indexes...")
        self.encoder = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)
        
        # Load metadata
        with open(index_directory / "metadata.json", 'r') as f:
            self.metadata = json.load(f)
        
        # Load BM25 index
        with open(index_directory / "bm25_index.pkl", 'rb') as f:
            self.bm25 = pickle.load(f)
            
        # Load FAISS index
        self.faiss_index = faiss.read_index(str(index_directory / "faiss_index.bin"))
        logging.info("Models and indexes loaded successfully.")

    def search(self, query: str, top_k: int = 5, bm25_k: int = 20, faiss_k: int = 20):
        """Performs the full hybrid search and reranking pipeline."""
        
        # --- 1. Sparse Search (BM25) ---
        tokenized_query = query.split(" ")
        bm25_scores = self.bm25.get_scores(tokenized_query)
        # Get top candidate indices from BM25
        bm25_indices = np.argsort(bm25_scores)[::-1][:bm25_k]

        # --- 2. Dense Search (FAISS) ---
        query_embedding = self.encoder.encode([query])
        # Get top candidate indices from FAISS
        _, faiss_indices = self.faiss_index.search(query_embedding, faiss_k)
        faiss_indices = faiss_indices[0] # Search returns a 2D array

        # --- 3. Fusion ---
        # Combine and deduplicate candidate indices
        candidate_indices = list(set(bm25_indices) | set(faiss_indices))
        logging.info(f"Found {len(candidate_indices)} unique candidates from sparse and dense search.")
        
        # --- 4. Reranking ---
        # Prepare pairs of [query, document_text] for the reranker
        rerank_pairs = [[query, self.metadata[i]['text']] for i in candidate_indices]
        
        if not rerank_pairs:
            return []

        logging.info("Reranking candidates with Cross-Encoder...")
        rerank_scores = self.reranker.predict(rerank_pairs)
        
        # Combine candidates with their new scores
        reranked_results = list(zip(candidate_indices, rerank_scores))
        # Sort by the new reranker score in descending order
        reranked_results.sort(key=lambda x: x[1], reverse=True)
        
        # --- 5. Final Top-K Results ---
        final_indices = [idx for idx, score in reranked_results[:top_k]]
        final_results = [self.metadata[i] for i in final_indices]
        
        return final_results

# --- Example Usage ---
if __name__ == "__main__":
    searcher = HybridSearcher(INDEX_DIR)
    
    query = "What is reinforcement learning?"
    
    print(f"Executing hybrid search for query: '{query}'")
    results = searcher.search(query)
    
    print("\n--- Top 5 Reranked Results ---")
    for i, doc in enumerate(results):
        print(f"{i+1}. {doc['source_title']}")
        print(f"   Excerpt: {doc['text'][:200]}...")
        print("-" * 20)

    # This output `results` is what you would feed into your LLM's context.