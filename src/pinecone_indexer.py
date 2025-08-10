import json
import os
from pathlib import Path
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
from tqdm import tqdm
import torch
import logging

# --- Configuration ---
# Load environment variables from .env file
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Path to your processed data
INPUT_FILE = Path("data/processed_data/processed_chunks.jsonl")

# Pinecone index settings
INDEX_NAME = "ai-wikipedia" # Must match the name of your index in Pinecone

# Embedding model - this model outputs 384-dimensional vectors
MODEL_NAME = 'all-MiniLM-L6-v2'

# Batch size for upserting to Pinecone
BATCH_SIZE = 100

def create_pinecone_index_if_not_exists(pc: Pinecone, index_name: str, dimension: int):
    """Checks if a Pinecone index exists and creates it if it doesn't."""
    if index_name not in pc.list_indexes().names():
        logging.info(f"Index '{index_name}' not found. Creating a new one...")
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws', # Or 'gcp'
                region='us-west-2' # Choose a region
            )
        )
        logging.info(f"Index '{index_name}' created successfully.")
    else:
        logging.info(f"Index '{index_name}' already exists.")

def main():
    """
    Main function to generate embeddings and upsert them to Pinecone.
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Validate environment variables
    if not PINECONE_API_KEY:
        logging.error("Pinecone API key not set in .env file.")
        return

    # 2. Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # 3. Initialize the embedding model
    # Check if a CUDA-enabled GPU is available, otherwise use CPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logging.info(f"Using device: {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)
    
    # The dimension must match what you set in the Pinecone dashboard
    embedding_dimension = model.get_sentence_embedding_dimension()
    logging.info(f"Embedding model loaded. Vector dimension: {embedding_dimension}")

    # 4. Ensure Pinecone index exists
    create_pinecone_index_if_not_exists(pc, INDEX_NAME, embedding_dimension)
    index = pc.Index(INDEX_NAME)

    # 5. Read data and upsert to Pinecone in batches
    logging.info(f"Reading data from {INPUT_FILE}...")
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        chunks = [json.loads(line) for line in f]

    logging.info(f"Starting upsert to Pinecone index '{INDEX_NAME}'...")
    for i in tqdm(range(0, len(chunks), BATCH_SIZE), desc="Upserting batches"):
        batch = chunks[i:i + BATCH_SIZE]
        
        # Get the text from each chunk in the batch
        texts = [chunk['text'] for chunk in batch]
        
        # Generate embeddings for the batch
        embeddings = model.encode(texts, convert_to_tensor=True, device=device).tolist()
        
        # Prepare vectors for upsert
        # Each vector needs an ID and its embedding. We also store metadata.
        vectors_to_upsert = []
        for j, chunk in enumerate(batch):
            vector = {
                "id": chunk['chunk_id'],
                "values": embeddings[j],
                "metadata": {
                    "text": chunk['text'], # Storing original text is useful
                    "source_title": chunk['source_title'],
                    "source_url": chunk['source_url'],
                    "chunk_number": chunk['chunk_number']
                }
            }
            vectors_to_upsert.append(vector)
        
        # Upsert the batch to Pinecone
        index.upsert(vectors=vectors_to_upsert)

    logging.info("Upsert process complete.")
    logging.info(f"Index stats: {index.describe_index_stats()}")


if __name__ == "__main__":
    main()