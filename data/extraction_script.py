import wikipediaapi
import json
import time
import os
from dotenv import load_dotenv

# This function searches for a .env file and loads its variables
# into the environment so os.getenv() can access them.
load_dotenv()

# Use os.getenv() to retrieve the value of your variable.
# It's good practice to provide a default value (like None) in case it's not found.
email_address = os.getenv("EMAIL")

# 1. Initialize the Wikipedia API
#    It's good practice to specify a user-agent
wiki_api = wikipediaapi.Wikipedia(
    language='en',
    user_agent=f"MyDocumentSearchProject/1.0 ({email_address})"
)

# 2. Define a function to get all article pages from a category
def get_pages_in_category(category_name, max_articles=200):
    """Recursively fetches articles from a Wikipedia category."""
    category_page = wiki_api.page(f"Category:{category_name}")
    pages = []

    # We use categorymembers to get sub-categories and pages
    for member in category_page.categorymembers.values():
        if len(pages) >= max_articles:
            break

        # If the member is a sub-category, recurse
        if member.namespace == wikipediaapi.Namespace.CATEGORY:
            print(f"Descending into sub-category: {member.title}")
            # Be careful with recursion depth to avoid getting too many articles
            # We'll do a shallow recursion for this example
            # pages.extend(get_pages_in_category(member.title.replace("Category:", ""), max_articles=max_articles - len(pages)))

        # If the member is an article page (namespace 0)
        elif member.namespace == wikipediaapi.Namespace.MAIN:
            print(f"Fetching article: {member.title}")
            pages.append({'title': member.title, 'text': member.text, 'url': member.fullurl})
            # Add a small delay to be polite to the API
            time.sleep(0.5)

    return pages

# 3. Choose a starting category and run the extraction
target_category = "Artificial intelligence"
print(f"Starting extraction for category: {target_category}")
articles = get_pages_in_category(target_category, max_articles=500)

# 4. Save the data to a file
#    JSON is a great format for this
with open(f"data/raw_data/{target_category.replace(" ", "_")}.json", "w", encoding="utf-8") as f:
    json.dump(articles, f, ensure_ascii=False, indent=4)

print(f"\nSuccessfully extracted {len(articles)} articles and saved to wikipedia_ai_corpus.json")