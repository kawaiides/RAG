import wikipediaapi
import json
import time
import logging
import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from pinecone import Pinecone, ServerlessSpec
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pathlib import Path
import re
from dotenv import load_dotenv
import requests

# --- Configuration ---
RAW_DATA_DIR = Path("data/raw_data")
LOCAL_INDEX_DIR = Path("data/local_hybrid_index")
EMBEDDING_MODEL = 'all-MiniLM-L6-v2'
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Ensure directories exist
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
LOCAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

def search_wiki_categories(search_term: str, limit: int = 7) -> list[str]:
    """Searches for Wikipedia categories using the MediaWiki API."""
    if not search_term:
        return []
    
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    
    PARAMS = {
        "action": "opensearch",
        "namespace": "14",  # Namespace 14 is for Categories
        "search": f"Category:{search_term}",
        "limit": str(limit),
        "format": "json"
    }
    
    try:
        response = S.get(url=URL, params=PARAMS)
        response.raise_for_status()
        data = response.json()
        # The result is a list like [searchTerm, [results], [descriptions], [links]]
        # We just need the results list, and we'll strip the "Category:" prefix.
        return [item.replace("Category:", "") for item in data[1]]
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to search Wikipedia categories: {e}")
        return []

# --- Part 1: Data Extraction ---
def extract_from_wikipedia(keyword: str, max_articles: int = 200) -> Path:
    """
    Extracts articles for a keyword from Wikipedia and saves them to a raw data file.
    Returns the path to the created raw data file.
    """
    logging.info(f"Starting Wikipedia extraction for keyword: '{keyword}'")
    
    email_address = os.getenv("EMAIL", "your-email@example.com") # Fetch email from .env
    wiki_api = wikipediaapi.Wikipedia(language='en', user_agent=f"MyAIResearchProject/1.0 ({email_address})")
    
    category_page = wiki_api.page(f"Category:{keyword}")
    if not category_page.exists():
        raise FileNotFoundError(f"Wikipedia Category page '{keyword}' does not exist.")
        
    articles = []
    
    def get_articles_recursive(category, current_articles, max_count, depth=0, max_depth=2):
        if depth >= max_depth or len(current_articles) >= max_count:
            return
        
        for member in category.categorymembers.values():
            if len(current_articles) >= max_count:
                break
            if member.namespace == wikipediaapi.Namespace.CATEGORY:
                get_articles_recursive(member, current_articles, max_count, depth + 1, max_depth)
            elif member.namespace == wikipediaapi.Namespace.MAIN:
                if member.title not in [a['title'] for a in current_articles]:
                    logging.info(f"Fetching article: {member.title}")
                    current_articles.append({'title': member.title, 'text': member.text, 'url': member.fullurl})
                    time.sleep(0.1)

    get_articles_recursive(category_page, articles, max_articles)

    if not articles:
        raise ValueError(f"No articles found for keyword '{keyword}'. Try a different keyword.")

    output_filename = RAW_DATA_DIR / f"{keyword.lower().replace(' ', '_')}.jsonl"
    with open(output_filename, "w", encoding="utf-8") as f:
        for article in articles:
            f.write(json.dumps(article) + "\n")
            
    logging.info(f"Extraction complete. Saved {len(articles)} articles to {output_filename}.")
    return output_filename

# --- Part 2: Preprocessing ---
def clean_text(text: str) -> str:
    text = re.sub(r'==.*?==', '', text)
    text = re.sub(r'{{.*?}}', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = '\n'.join([line.strip() for line in text.split('\n')])
    return text.strip()

def preprocess_and_chunk(raw_data_path: Path) -> list[dict]:
    """Preprocesses a raw data file and returns a list of chunked documents."""
    logging.info(f"Preprocessing and chunking data from {raw_data_path}...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len
    )

    processed_chunks = []
    with open(raw_data_path, 'r', encoding='utf-8') as f:
        articles = [json.loads(line) for line in f]

    for article in articles:
        cleaned_text = clean_text(article['text'])
        chunks = text_splitter.split_text(cleaned_text)
        for i, chunk_text in enumerate(chunks):
            processed_chunks.append({
                'chunk_id': f"{article['title']}_{i}",
                'source_title': article['title'],
                'source_url': article['url'],
                'text': chunk_text
            })
    logging.info(f"Created {len(processed_chunks)} chunks.")
    return processed_chunks

# --- Part 3: Local Indexing ---
def build_local_indexes_for_keyword(chunks: list[dict], index_name: str):
    """Builds and saves local BM25 and FAISS indexes for a given set of chunks."""
    index_path = LOCAL_INDEX_DIR / index_name
    index_path.mkdir(parents=True, exist_ok=True)
    
    corpus = [doc['text'] for doc in chunks]

    # BM25 Index
    logging.info(f"Building BM25 index for '{index_name}'...")
    tokenized_corpus = [doc.split(" ") for doc in corpus]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(index_path / "bm25_index.pkl", 'wb') as f:
        pickle.dump(bm25, f)

    # FAISS Index
    logging.info(f"Building FAISS index for '{index_name}'...")
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = encoder.encode(corpus, show_progress_bar=True, convert_to_numpy=True)
    
    embedding_dim = embeddings.shape[1]
    index = faiss.IndexIDMap(faiss.IndexFlatL2(embedding_dim))
    ids = np.array(range(len(corpus)))
    index.add_with_ids(embeddings, ids)
    faiss.write_index(index, str(index_path / "faiss_index.bin"))

    # Metadata
    with open(index_path / "metadata.json", 'w', encoding='utf-8') as f:
        json.dump(chunks, f)
    logging.info(f"Local index '{index_name}' built successfully.")

# --- Part 4: Pinecone Indexing ---
def index_to_pinecone_for_keyword(chunks: list[dict], index_name: str, api_key: str, batch_size: int = 100):
    """Generates embeddings and upserts them to a specific Pinecone index."""
    logging.info(f"Starting Pinecone indexing for index '{index_name}'...")
    pc = Pinecone(api_key=api_key)
    
    encoder = SentenceTransformer(EMBEDDING_MODEL)
    embedding_dim = encoder.get_sentence_embedding_dimension()

    if index_name not in pc.list_indexes().names():
        logging.info(f"Pinecone index '{index_name}' not found. Creating new index...")
        pc.create_index(
            name=index_name,
            dimension=embedding_dim,
            metric='cosine',
            spec=ServerlessSpec(cloud='aws', region='us-west-2')
        )
        time.sleep(1) # Wait for index to be ready

    index = pc.Index(index_name)
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        texts = [chunk['text'] for chunk in batch]
        ids = [chunk['chunk_id'] for chunk in batch]
        
        embeddings = encoder.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        
        vectors_to_upsert = []
        for j, chunk in enumerate(batch):
            vectors_to_upsert.append({
                "id": ids[j],
                "values": embeddings[j].tolist(),
                "metadata": {
                    "text": chunk['text'],
                    "source_title": chunk['source_title'],
                    "source_url": chunk['source_url']
                }
            })
        index.upsert(vectors=vectors_to_upsert)
        logging.info(f"Upserted batch {i//batch_size + 1} to Pinecone.")
    logging.info(f"Pinecone index '{index_name}' updated successfully.")