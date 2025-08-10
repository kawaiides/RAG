import wikipediaapi
import json
import time
import logging
import os
from dotenv import load_dotenv

# This function searches for a .env file and loads its variables
# into the environment so os.getenv() can access them.
load_dotenv()

# Use os.getenv() to retrieve the value of your variable.
# It's good practice to provide a default value (like None) in case it's not found.
email_address = os.getenv("EMAIL")

# Configure logging to see progress and errors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_articles_from_category(category_name: str, max_articles: int = 500, max_depth: int = 2, current_depth: int = 0):
    """
    Recursively fetches articles from a Wikipedia category and its subcategories.
    
    Args:
        category_name (str): The starting category (e.g., "Artificial intelligence").
        max_articles (int): The maximum number of articles to fetch.
        max_depth (int): How many levels of subcategories to explore.
        current_depth (int): The current recursion depth.
    
    Returns:
        A list of article dictionaries.
    """
    if current_depth >= max_depth:
        logging.info(f"Max depth of {max_depth} reached. Stopping recursion here.")
        return []

    # Initialize the API with a descriptive user-agent
    wiki_api = wikipediaapi.Wikipedia(
        language='en',
        user_agent=f"MyAIResearchProject/1.0 ({email_address})"
    )
    
    category_page = wiki_api.page(f"Category:{category_name}")
    if not category_page.exists():
        logging.error(f"Category page '{category_name}' does not exist.")
        return []
        
    articles = []
    
    # Iterate through all members of the category
    for member in category_page.categorymembers.values():
        if len(articles) >= max_articles:
            break
            
        # If it's a subcategory, recurse deeper
        if member.namespace == wikipediaapi.Namespace.CATEGORY:
            logging.info(f"Descending into subcategory: {member.title}")
            sub_articles = get_articles_from_category(
                member.title.replace("Category:", ""),
                max_articles - len(articles),
                max_depth,
                current_depth + 1
            )
            articles.extend(sub_articles)
            
        # If it's an article page, grab its content
        elif member.namespace == wikipediaapi.Namespace.MAIN:
            try:
                logging.info(f"Fetching article: {member.title}")
                page_content = member.text
                if page_content: # Ensure page is not empty
                    articles.append({
                        'title': member.title,
                        'text': page_content,
                        'url': member.fullurl
                    })
                # Be polite with a small delay
                time.sleep(0.1) 
            except Exception as e:
                logging.error(f"Could not fetch article {member.title}. Error: {e}")

    return articles

# --- Main Execution ---
if __name__ == "__main__":
    target_category = "Artificial intelligence"
    corpus_articles = get_articles_from_category(target_category, max_articles=1000, max_depth=3)
    
    # Save to a JSON Lines (.jsonl) file, which is better for large datasets
    # Each line is a separate JSON object
    output_filename = f"data/raw_data/{target_category.replace(" ", "_")}.jsonl"
    with open(output_filename, "w", encoding="utf-8") as f:
        for article in corpus_articles:
            f.write(json.dumps(article) + "\n")
            
    logging.info(f"Extraction complete. Saved {len(corpus_articles)} articles to {output_filename}.")