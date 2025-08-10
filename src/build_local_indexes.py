import json
from pathlib import Path
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import logging

# --- Configuration ---
INPUT_FILE = Path("data/processed_data/processed_chunks.jsonl")
INDEX_DIR = Path("data/local_hybrid_index")
MODEL_NAME = 'all-MiniLM-L6-v2'

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def build_indexes():
    """Reads processed data and builds BM25, FAISS, and metadata files."""
    logging.info("Starting to build local indexes...")
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load the processed data
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        documents = [json.loads(line) for line in f]
    corpus = [doc['text'] for doc in documents]
    logging.info(f"Loaded {len(corpus)} document chunks.")

    # --- 2. Build BM25 Index ---
    logging.info("Building BM25 index...")
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    
    # Save the BM25 index using pickle
    with open(INDEX_DIR / "bm25_index.pkl", 'wb') as f:
        pickle.dump(bm25, f)
    logging.info("BM25 index saved successfully.")

    # --- 3. Build FAISS Index ---
    logging.info("Building FAISS index...")
    encoder = SentenceTransformer(MODEL_NAME)
    embeddings = encoder.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))
    
    # We use the document's original index as its ID in FAISS
    ids = np.array(range(len(corpus)))
    index.add_with_ids(embeddings, ids)
    
    faiss.write_index(index, str(INDEX_DIR / "faiss_index.bin"))
    logging.info("FAISS index saved successfully.")
    
    # --- 4. Save Metadata ---
    # We just need the original documents list for metadata lookup
    with open(INDEX_DIR / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(documents, f)
    logging.info("Metadata file saved.")

    logging.info("All indexes built and saved.")

if __name__ == "__main__":
    build_indexes()