import json
import re
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
import uuid
from tqdm import tqdm

# --- Configuration ---
# Set up paths using pathlib for robust path handling
INPUT_FILE = Path("data/raw_data/Artificial_intelligence.jsonl")
OUTPUT_DIR = Path("data/processed_data")
OUTPUT_FILE = OUTPUT_DIR / "processed_chunks.jsonl"

# Text splitting parameters
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# Setup basic logging to see the script's progress and any issues
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def clean_text(text: str) -> str:
    """
    Cleans the raw text from Wikipedia by removing common artifacts.
    - Removes Wikipedia-style section headers (e.g., == History ==)
    - Removes excessive newlines and whitespace
    - Removes templates and other specific markup remnants
    """
    # Remove section headers like == See also ==
    text = re.sub(r'==.*?==', '', text)
    # Remove templates like {{...}}
    text = re.sub(r'{{.*?}}', '', text)
    # Normalize whitespace: replace multiple newlines with two (paragraph break)
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove leading/trailing whitespace from each line
    text = '\n'.join([line.strip() for line in text.split('\n')])
    return text.strip()


def preprocess_and_chunk_data():
    """
    Main function to read, clean, chunk, and save the data.
    """
    logging.info("Starting data preprocessing...")

    # 1. Ensure input file exists
    if not INPUT_FILE.is_file():
        logging.error(f"Input file not found at: {INPUT_FILE}")
        return

    # 2. Ensure the output directory exists, create it if it doesn't
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output directory '{OUTPUT_DIR}' is ready.")

    # 3. Initialize the text splitter from LangChain
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        add_start_index=True, # Helpful for context, though we won't use it directly here
    )

    # 4. Read the raw data, process, and write to the output file
    processed_chunks = []
    
    # Read the number of lines for the progress bar
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        num_lines = sum(1 for line in f)

    logging.info(f"Reading and processing {num_lines} articles from {INPUT_FILE}...")
    
    with open(INPUT_FILE, 'r', encoding='utf-8') as infile:
        # Use tqdm for a nice progress bar
        for line in tqdm(infile, total=num_lines, desc="Processing articles"):
            article = json.loads(line)
            
            title = article.get('title')
            text = article.get('text')
            url = article.get('url')
            
            if not text or not title:
                continue
                
            # Clean the text first
            cleaned_text = clean_text(text)
            
            # Split the cleaned text into chunks
            chunks = text_splitter.split_text(cleaned_text)
            
            # Create a structured record for each chunk
            for i, chunk_text in enumerate(chunks):
                chunk_record = {
                    'chunk_id': str(uuid.uuid4()),  # A unique ID for each chunk
                    'source_title': title,
                    'source_url': url,
                    'chunk_number': i + 1,
                    'text': chunk_text
                }
                processed_chunks.append(chunk_record)

    # 5. Save the processed chunks to the output file
    logging.info(f"Saving {len(processed_chunks)} processed chunks to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as outfile:
        for chunk in tqdm(processed_chunks, desc="Saving chunks"):
            outfile.write(json.dumps(chunk) + '\n')

    logging.info("Data preprocessing complete!")


if __name__ == "__main__":
    preprocess_and_chunk_data()