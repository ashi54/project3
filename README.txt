Web Crawler and Search Engine

The M3 project is about a web crawler and search engine that processes a dataset of 
ICS web pages, extracts and indexes their content, and enables efficient searching 
using TF-IDF ranking. There are 3 parts to the project:

-> A crawler and indexer that extracts text from JSON files containing HTML content 
   and creates an inverted index.
-> A search interface that allows users to input queries and retrieve relevant web 
   pages.
-> A Flask-based web application that provides a user-friendly interface for 
   searching.

__________________________________________________________________________________

Project Structure

/crawler_web
│── Crawler_logic.py          # Core crawling & indexing logic
│── main.py                   # Entry point for the Flask application
│── final_inverted_index.json  # Merged index file (not included in submission)
│── partial_index_1.json       # Partial index 1 for large datasets
│── partial_index_2.json       # Partial index 2 for large datasets
│── partial_index_3.json       # Partial index 3 for large datasets
│── partial_index_4.json       # Partial index 4 for large datasets
│── partial_index_6.json       # Partial index 5 for large datasets
│── partial_index_6.json       # Partial index 6 for large datasets
│── DEV/                       # Folder containing JSON web pages (not included in submission)
│── website/
│   ├── __init__.py            # Initializes the Flask app
│   ├── views.py               # Handles routing and search logic
│   ├── templates/
│       ├── base.html          # HTML template for the search page
│── __pycache__/               # Cached Python files (ignore)
│── README.txt                 # Instructions on how to run the project
│── TEST.txt                   # Contains test queries and analysis

__________________________________________________________________________________

How the Search Works

1. The user enters a search query in the web UI.
2. The query is tokenized and stemmed before searching.
3. The system performs a Boolean search based on an inverted index.
4. Results are ranked using TF-IDF weighting to improve relevance.
5. The top 5 most relevant URLs are displayed.
