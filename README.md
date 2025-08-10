Document Search and Summarization using LLMs
============================================

This project implements a sophisticated Retrieval-Augmented Generation (RAG) system to perform efficient search and summarization on a large corpus of text. It leverages a hybrid search approach, combining traditional keyword-based search (BM25) with modern semantic search (FAISS), and uses a Large Language Model (LLM) like Google's Gemini to generate accurate, source-cited summaries. The system is accessible through an interactive Streamlit application that supports both a fully local backend and a cloud-based backend using Pinecone.

* * * * *

Features
--------

-   **End-to-End RAG Pipeline:** From raw data extraction to final summary generation.

-   **Hybrid Search:** Combines the strengths of keyword (BM25) and semantic (vector) search for superior retrieval accuracy.

-   **Cross-Encoder Reranking:** Employs a powerful reranker model to refine search results before generation.

-   **LLM-Powered Summarization:** Uses Google's Gemini to generate coherent and factually grounded summaries.

-   **Source Citations:** Automatically includes links to the source documents used for the summary, ensuring verifiability.

-   **Interactive Interface:** A Streamlit application provides a user-friendly way to interact with the system.

-   **Dual Backend Support:** The interface can switch between a fully local search pipeline and a cloud-based pipeline using Pinecone.

-   **Automated Evaluation:** Includes a script to benchmark the performance of both pipelines using ROUGE scores.

* * * * *

System Architecture
-------------------

The system is built on a Retrieval-Augmented Generation (RAG) architecture, which consists of two main stages:

1.  **Offline Indexing:** This is a one-time process where the source documents (e.g., Wikipedia articles) are cleaned, split into manageable chunks, and indexed. Two types of indexes are created:

    -   A **BM25 index** for efficient keyword-based retrieval.

    -   A **FAISS vector index** (for local search) or a **Pinecone index** (for cloud search) for fast semantic similarity search based on embeddings.

2.  **Online Inference:** This stage is triggered when a user submits a query via the Streamlit app.

    -   **Retrieve:** The system performs a search using the selected backend (local hybrid or Pinecone) to retrieve a set of candidate documents.

    -   **Rerank (Local Pipeline):** The local pipeline uses a Cross-Encoder model to rerank these candidates to find the most relevant ones.

    -   **Generate:** The top-ranked documents are passed as context to an LLM, which generates a final summary or answer based on the provided information.

* * * * *

Setup and Installation
----------------------

Follow these steps to set up and run the project locally.

### 1\. Clone the Repository

Bash

```
git clone <your-repository-url>
cd document-search-summarization

```

### 2\. Create a Virtual Environment

It's highly recommended to use a virtual environment to manage dependencies.

Bash

```
python -m venv myenv
source myenv/bin/activate  # On Windows, use `myenv\Scripts\activate`

```

### 3\. Install Dependencies

Install all the required Python libraries from the `requirements.txt` file.

Bash

```
pip install -r requirements.txt

```

### 4\. Configure Environment Variables

Create a `.env` file in the root directory of the project by copying the example below. This file will store your secret API keys.

Code snippet

```
# .env

# Required for both pipelines to generate answers and test questions
GOOGLE_API_KEY="YOUR_GOOGLE_GEMINI_API_KEY"

# Required only for the Pinecone pipeline
PINECONE_API_KEY="YOUR_PINECONE_API_KEY"
PINECONE_ENVIRONMENT="your-pinecone-environment-name" # e.g., gcp-starter

# Required for the data extraction script
EMAIL="your-email@example.com"

```

* * * * *

Usage Workflow
--------------

The system is run in three main steps: data preparation, index building, and running the application.

### Step 1: Data Preparation

First, you need a corpus of documents. The `src/data_management.py` script can extract data from Wikipedia for a given category. For this example, we'll assume you are creating an index on "Artificial intelligence".

1.  **Extract Data:** Use the functions within `src/data_management.py` to extract articles. This will create a file like `data/raw_data/artificial_intelligence.jsonl`.

2.  **Pre-process Data:** The data management script also handles cleaning the raw text and splitting it into chunks suitable for indexing, creating a file like `data/processed_data/processed_chunks.jsonl`.

### Step 2: Building the Indexes

You can build indexes for either the local pipeline or the Pinecone pipeline.

-   For the Local Hybrid Search Pipeline:

    Run the build_local_indexes.py script. This will create BM25 and FAISS indexes from your processed chunks and save them to data/local_hybrid_index/.

    Bash

    ```
    python src/build_local_indexes.py

    ```

-   For the LangChain + Pinecone Pipeline:

    Run the pinecone_indexer.py script. This will generate embeddings and upload them to your specified Pinecone index. Make sure you have created an index in your Pinecone account first.

    Bash

    ```
    python src/pinecone_indexer.py

    ```

### Step 3: Running the Application

Once your data is prepared and indexed, launch the Streamlit interface.

Bash

```
streamlit run app.py

```

Navigate to the URL provided by Streamlit in your browser. You can now select your desired pipeline, choose an index, and start asking questions!

* * * * *

Codebase Overview
-----------------

-   `src/data_management.py`: A centralized module containing functions to extract data from Wikipedia, preprocess it, and build indexes for both local and Pinecone backends.

-   `src/local_hybrid_search.py`: Contains the `HybridSearcher` class which implements the core logic for combining BM25 and FAISS search results, followed by reranking.

-   `src/summarizer.py`: Orchestrates the local pipeline by using the `HybridSearcher` and an LLM to generate a final answer with source citations.

-   `src/langchain_query_engine.py`: Implements the same RAG logic but using the LangChain framework for a more modular pipeline connected to Pinecone.

-   `src/evaluation.py`: A script to automatically evaluate and compare the performance of the two RAG pipelines using ROUGE scores.

-   `app.py`: The main Streamlit application that provides the user interface, allowing users to switch between pipelines, select indexes, and query the system.

* * * * *

Evaluation
----------

The `src/evaluation.py` script provides an automated way to benchmark the two pipelines. To run it:

Bash

```
python src/evaluation.py

```

This script will:

1.  Randomly sample 30 chunks from your processed data.

2.  Use the Gemini API to generate a relevant question for each chunk.

3.  Run each question through both the **LangChain + Pinecone** and **Local Hybrid Search** pipelines.

4.  Calculate the ROUGE-1, ROUGE-2, and ROUGE-L F1-scores by comparing each pipeline's output to the original text chunk.

5.  Print a summary table comparing the average scores, allowing for a direct performance comparison.