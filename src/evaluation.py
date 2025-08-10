import os
import json
import random
import logging
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

# --- ROUGE Scorer ---
from rouge_score import rouge_scorer

# --- LangChain for Question Generation ---
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Import your RAG Pipeline Components ---
# NOTE: These imports assume your pipeline scripts are modified to accept an `index_name`.
from langchain_query_engine import initialize_components as init_langchain, create_rag_chain_with_sources, format_final_output
from summarizer import SummarizationPipeline

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

PROCESSED_DATA_FILE = "data/processed_data/processed_chunks.jsonl"
NUM_SAMPLES = 30
# Use a default index name, assuming your data is indexed under this name.
# This should match the index you built with your scripts.
DEFAULT_INDEX_NAME = "ai-wikipedia" 

# --- Initialize Gemini for Question Generation using LangChain ---
try:
    # This model will be used specifically for generating questions
    question_generation_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
except Exception as e:
    logging.error(f"Failed to initialize Google Gemini for question generation: {e}")
    question_generation_llm = None

def load_and_sample_data(filepath: str, num_samples: int) -> list[dict]:
    """Loads processed chunks and returns a random sample."""
    logging.info(f"Loading data from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        all_chunks = [json.loads(line) for line in f]
    
    if len(all_chunks) < num_samples:
        raise ValueError(f"Not enough chunks in the data file to sample {num_samples}. Found only {len(all_chunks)}.")
        
    return random.sample(all_chunks, num_samples)

def generate_question_with_llm(chunk_text: str, llm) -> str:
    """Uses a LangChain LLM to generate a question that the chunk_text can answer."""
    if not llm:
        raise ConnectionError("Question generation LLM not initialized.")
        
    prompt = f"""
    Based on the following text, generate a single, concise, and clear question that this text can directly answer.
    Do not ask a question that requires information outside of this text.

    TEXT:
    ---
    {chunk_text}
    ---

    QUESTION:
    """
    try:
        response = llm.invoke(prompt)
        return response.content.strip()
    except Exception as e:
        logging.error(f"Error generating question with LLM: {e}")
        return f"What does the following text describe: '{chunk_text[:50]}...'?" # Fallback question

def run_evaluation():
    """Main function to run the end-to-end evaluation."""
    
    # 1. Load sample data
    test_samples = load_and_sample_data(PROCESSED_DATA_FILE, NUM_SAMPLES)
    reference_summaries = [sample['text'] for sample in test_samples]

    # 2. Generate questions for each sample
    logging.info(f"Generating {NUM_SAMPLES} test questions using Gemini...")
    questions = [generate_question_with_llm(text, question_generation_llm) for text in tqdm(reference_summaries, desc="Generating Questions")]

    # --- Initialize Pipelines ---
    logging.info("Initializing RAG pipelines...")
    # LangChain + Pinecone
    lc_llm, lc_vectorstore = init_langchain()
    langchain_pipeline = create_rag_chain_with_sources(lc_llm, lc_vectorstore)
    
    # Local Hybrid Search
    # NOTE: This assumes your SummarizationPipeline is modified to take `index_name`
    local_hybrid_pipeline = SummarizationPipeline()

    # 3. Get generated summaries from both pipelines
    logging.info("Generating summaries from both pipelines...")
    lc_generated_summaries = [format_final_output(langchain_pipeline.invoke({"question": q})) for q in tqdm(questions, desc="LangChain+Pinecone")]
    local_generated_summaries = [local_hybrid_pipeline.summarize(q) for q in tqdm(questions, desc="Local Hybrid Search")]

    # 4. Calculate ROUGE scores
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    def calculate_scores(generated, reference):
        scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
        for gen, ref in zip(generated, reference):
            score = scorer.score(ref, gen)
            scores['rouge1'].append(score['rouge1'].fmeasure)
            scores['rouge2'].append(score['rouge2'].fmeasure)
            scores['rougeL'].append(score['rougeL'].fmeasure)
        return {key: sum(value) / len(value) for key, value in scores.items()}

    logging.info("Calculating ROUGE scores...")
    lc_avg_scores = calculate_scores(lc_generated_summaries, reference_summaries)
    local_avg_scores = calculate_scores(local_generated_summaries, reference_summaries)

    # 5. Report Results
    report_data = {
        "Metric": ["ROUGE-1 F1", "ROUGE-2 F1", "ROUGE-L F1"],
        "LangChain + Pinecone": [
            lc_avg_scores['rouge1'],
            lc_avg_scores['rouge2'],
            lc_avg_scores['rougeL']
        ],
        "Local Hybrid Search": [
            local_avg_scores['rouge1'],
            local_avg_scores['rouge2'],
            local_avg_scores['rougeL']
        ]
    }
    
    df_report = pd.DataFrame(report_data)
    
    print("\n" + "="*80)
    print("--- RAG Pipeline Performance Evaluation ---")
    print(df_report.to_string(index=False))
    print("="*80)
    print("\n**Analysis:**")
    print("- The table above shows the average F1-scores for each ROUGE metric.")
    print("- Higher scores indicate better performance, meaning the generated summary is closer to the reference text.")
    print("- Compare the columns to see which retrieval strategy (cloud-based semantic vs. local hybrid) performed better on this dataset.")


if __name__ == "__main__":
    run_evaluation()
