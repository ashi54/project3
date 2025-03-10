import json
from bs4 import BeautifulSoup # O(n) bc you are parsing HTML, where n is the length of the content
import re  # O(n) -> n is the length of input string
from nltk.stem import PorterStemmer # O(n) where n is the length of the word being stemmed
from collections import defaultdict # O(1) checking the key and value
import os
import math
import time
import re

'''
Milestone 3 Objectives:
- Extract useful text from JSON files containing HTML content.
- Tokenize and normalize the extracted text.
- Store the inverted index on disk instead of memory to handle large datasets.
- Implement partial indexing and merge partial indexes.
- Perform Boolean and ranked TF-IDF search efficiently.
'''

stemmer = PorterStemmer()
doc_id_to_url = {}  
doc_lengths = defaultdict(float)  


def read_json_file(filepath):
    '''
    Here, we are extracting the "content" information inside each json file 
    because that is what we want for the inverted index.

    Time Complexity:
    - O(n): first read the file; n is the size of the file content (traversing it once)
    - O(n): parsing the JSON file since m represents the size of the JSON content
    - Total: O(n)
    '''
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
            return data.get("content", ""), data.get("url", "")
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return "", ""


def extract_text_from_html(htmlContent):
    '''
    After extracting the HTML from the json file, we need to extract the USEFUL text 
    from the HTML content of the webpage. 

    - What is considered useful? Article content, paragraphs, title, headings, bold text
    - What is not considered useful? <script>, <style>, home, contact us, buy now, copyright notices, etc.

    Return:
        resultDict dictionary that has the text and keyText

    Time Complexity:
    - O(n): parsing HTML (traversing through the size of the HTML content)
    - O(n): for the soup implementation and to remove the script and style
    - Total: O(n + n) = big O(n)
    '''
    try:
        soup = BeautifulSoup(htmlContent, 'html.parser')

        for tag in soup(["script", "style"]):  
            tag.extract()

        text = soup.get_text(separator=" ")  
        key_elements = []

        if soup.title and soup.title.string:
            key_elements.append(soup.title.string.strip())

        headings = []
        for h in soup.find_all(["h1", "h2", "h3"]):
            heading_text = h.get_text(separator=" ").strip()
            headings.append(heading_text)

        bold_text = []
        for b in soup.find_all(["b", "strong"]):
            bold_text_content = b.get_text(separator=" ").strip()
            bold_text.append(bold_text_content)

        key_elements.extend(headings)
        key_elements.extend(bold_text)

        key_text = " ".join(filter(None, key_elements)).strip()  

        return text.strip(), key_text
    except Exception as e:
        print(f"Error extracting text from HTML: {e}")
        return "", ""


def tokenize_and_normalize(text):
    '''
    Got the text and keyText and now we need to tokenize the input text into words, 
    convert them to lowercase, and apply porter stemming to reduce the words from the root form

    Return:
        Get the list of processed tokens after using the porter stemming.

    Time Complexity:
    - O(n): regex tokenization where n is the length of the input text
    - O(n): porter stemming where n is the length of each word
    - Total: O(n + n) = big O(n)
    '''
    tokens = re.findall(r'\b\w+\b', text.lower())  
    stemmed_tokens = []
    for token in tokens:
        stemmed_token = stemmer.stem(token)
        stemmed_tokens.append(stemmed_token)

    return stemmed_tokens


    # tokens = re.findall(r'\b\w+\b', text.lower())  # O(n)
    # return [stemmer.stem(token) for token in tokens]


def build_inverted_index(doc_id, tokens, key_tokens, inverted_index):
    '''
    Here, I am constructing the inverted index for a given document. This method counts 
    the occurrences of each token, calculates the term frequency (TF), and applies 
    TF-IDF weighting to improve ranking accuracy. Words that appear in important 
    sections(titles, headings, bold text) are given higher weight.

    doc_id = "/DEV/evoke_ics_uci_edu/page1.json"
    tokens = ['hello', 'world', 'hello']
    inverted_index = 
        {
        'hello': {'doc1.json': 2},
        'world': {'doc1.json': 1}
        }

    Time Complexity:
    - O(n): counting term frequencies (where n is the number of tokens in the document)
    - O(1): updating the inverted index for each token
    - O(n): calculating document length for normalization
    - Total: O(n + 1 + n) = O(n)
    '''
    term_frequency = defaultdict(int)

    for token in tokens:
        term_frequency[token] += 1  

    if not term_frequency:  
        return  

    max_tf = max(term_frequency.values()) 

    for token, freq in term_frequency.items():
        '''
        calculating the tf idf value
        '''
        tf = 0.5 + (0.5 * freq / max_tf)  

        if token in key_tokens:
            tf *= 2  

        inverted_index[token][doc_id] += tf

    sum_of_squares = 0
    for tf in term_frequency.values():
        sum_of_squares += tf ** 2

    doc_lengths[doc_id] = math.sqrt(sum_of_squares)


def save_partial_index(inverted_index, part_num):
    '''
    This method saves a portion of the inverted index into a json file on a disk and returns 
    the filename of the saved partial index. Using part_num as the part number representing
    the counter for which part the inverted index is being saved (prevents memory overflow)
    Eg. partial_index_1.json, partial_index_2.json, etc.

    inverted_index = 
        {
        'hello': {'doc1.json': 2},
        'world': {'doc1.json': 1}
        }
    part_num = 1

    Time Complexity:
    O(n): converting to a dictionary where the n is the number of tokens
    O(n): writing the file where n is the size of the output file
    Total: O(n + n) = big O(n)
    '''
    output_filename = f"partial_index_{part_num}.json"

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(inverted_index, f, indent=4)

    print(f"\nPartial index saved: {output_filename}")
    return output_filename


def process_all_documents(directory):
    '''
    This is for processing all of the json files in the DEV folder and calling all the methods 
    so everything will process at once. Making sure that if there is more than 10000 documents 
    processed, the current inverted index would be saved as the partial index and the index 
    would be cleared to save memory. Calling root (DEV) directory that contains ALL the json 
    files.

    Time complexity:
    O(n): traversing through the first n number of JSON files
    O(n): traversing through each file in the directory
    Total: O(n * n) 
        - Because there are 2 nested for loops, that means that the big o complexity is O(n * n)
    '''
    doc_count = 0
    part_num = 1
    partial_files = []
    inverted_index = defaultdict(lambda: defaultdict(float))  

    for root, _, files in os.walk(directory):  
        for file in files:  
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                content, url = read_json_file(file_path)
                if not content:
                    continue

                text, key_text = extract_text_from_html(content)
                tokens = tokenize_and_normalize(text)
                key_tokens = tokenize_and_normalize(key_text)

                build_inverted_index(file_path, tokens, key_tokens, inverted_index)
                doc_id_to_url[file_path] = url
                doc_count += 1

                # printing every 500 processed documents 
                if doc_count % 500 == 0:
                    print(f"Processed {doc_count} documents..")
                
                if doc_count % 10000 == 0:  
                    partial_filename = save_partial_index(inverted_index, part_num)
                    partial_files.append(partial_filename)
                    part_num += 1
                    # need to free memory for the current inverted index
                    inverted_index.clear()  

    if inverted_index:
        partial_filename = save_partial_index(inverted_index, part_num)
        partial_files.append(partial_filename)

    print(f"\nTotal Processed Documents: {doc_count}")
    return partial_files  


def merge_partial_indexes(partial_files, output_filename="final_inverted_index.json"):
    '''
    Here, I am merging all of the partial indexes we got from before and created a final 
    inverted index file. The partial indexes were created by batches to free up memory 
    and it makes sure that it is saved to the final json file.

    Time Complexity:
    O(n): traversing through the first partial index files
    O(n): traversing through the number of tokens and posting within each partial index file
    Total: O(n * n)
        - Because there are 2 nested for loops, that means that the big o complexity is O(n * n)
    '''
    final_index = defaultdict(lambda: defaultdict(float))

    for partial in partial_files:
        with open(partial, "r", encoding="utf-8") as f:
            partial_index = json.load(f)
            for token, postings in partial_index.items():
                for doc_id, tf in postings.items():
                    final_index[token][doc_id] += tf  

    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(final_index, f, indent=4)

    print(f"\nFinal merged index saved as {output_filename}")
    return final_index

def Bool_Search(queries, inverted_index):
    '''
    Here, I am performing a Boolean search on the inverted index. Given a query, the function 
    finds all documents that contain the query terms. It normalizes and tokenizes the query, 
    then looks up the index to retrieve relevant document sets. The results are computed 
    using set intersection, ensuring that only documents containing all query terms are 
    returned.

    Time Complexity:
    O(n): tokenizing and normalizing the query terms
    O(n): retrieving the document sets from the inverted index
    O(n): performing set intersection on the retrieved sets
    Total: O(n + n + n) = O(n)
    '''
    tokens = tokenize_and_normalize(queries)
    print("Search tokens:", tokens)

    if not tokens:
        return set()

    doc_sets = [set(inverted_index.get(token, {}).keys()) for token in tokens if token in inverted_index]

    if not doc_sets:
        return set()  

    result_docs = set.intersection(*doc_sets) if doc_sets else set()

    return result_docs


def print_top_5_urls(result_docs, inverted_index, query):
    '''
    This function ranks and retrieves the top 5 most relevant URLs based on a search query. 
    The ranking is determined using **TF-IDF scores**, ensuring the most meaningful matches 
    appear first.

    Time Complexity:
    - O(n): Tokenizing and normalizing the query (where n is the query length).
    - O(m): Summing TF-IDF scores for all matched documents (where m is the number of docs).
    - O(m log m): Sorting the documents by score.
    - Total: O(m log m)** (due to sorting step).
    '''
    query_tokens = tokenize_and_normalize(query)

    ranked_docs = sorted(
        result_docs,
        key=lambda doc: sum(inverted_index[token].get(doc, 0) for token in query_tokens if token in inverted_index),
        reverse=True
    )

    top_docs = ranked_docs[:5]

    top_urls = [doc_id_to_url.get(doc_id, "URL Not Found") for doc_id in top_docs]

    return top_urls
