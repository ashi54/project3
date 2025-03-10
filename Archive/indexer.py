import json
from bs4 import BeautifulSoup # O(n) bc you are parsing HTML, where n is the length of the content
import re # O(n) -> n is the length of input string
from nltk.stem import PorterStemmer # O(n) where n is the length of the word being stemmed
from collections import defaultdict # O(1) checking the key and value
import os

'''
Milestone 1 Objectives:

We need to extract useful text content from each webpage (json file), 
break it into searchable words, organize the information into the giant map 
(inverted index) and finally see which documents contain that specific word.

Inside DEV folder:
-> DEV
    -> evoke_ics...
        -> 0103.json
        -> 0292.json
    -> aiclub_ics...

The json file stores the URL, content, and encoding. We want the content which 
stores all the information in the HTML format. HTML gives an organized structure 
of the content inside the web page -> heading, paragraphs, images, tables, links, 
etc. We want to first extract the information in content because that is how we 
will make the inverted index.
'''

stemmer = PorterStemmer()

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
            # specifically need to get the content section inside the .json file
            return data.get("content", "")
    # error case
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return ""


def extract_text_from_html(htmlContent):
    '''
    After extracting the HTML from the json file, we need to extract the USEFUL text 
    from the HTML content of the webpage. 
    
    - What is considered useful? Article content, paragraphs, title, headings, bold text
    - What is not considered useful? <script>, <style>, home, contact us, buy now, copyright 
                                     notices, etc.
    
    Return:
        resultDict dictionary that has the text and keyText

    Time Complexity:
    - O(n): parsing HTML (traversing through the size of the HTML content)
    - O(n): for the soup implementation and to remove the script and style
    - Total: O(n + n) = big O(n)
    '''
    try:

        # using beautifulsoup object to parse the HTML
        # beautifulsoup is used to specifically parse HTML documentts
        soup = BeautifulSoup(htmlContent, 'html.parser') # O(n)

        # need to remove script and style since no useful info
        for tag in soup(["script", "style"]): # O(n)
            tag.extract()

        # extract the visible text and using seperator to seperate it with spaces
        text = soup.get_text(separator=" ") # O(n)
        # print(text)

        keyElements = []
        title = ""

        if soup.title and soup.title.string:
            # need to trip the page title
            title = soup.title.string.strip() # O(1)
        
        # need to extract all the headings (h1, h2, h3) for the main sections
        headings = []
        for h in soup.find_all(["h1", "h2", "h3"]): # O(n)
            # calling strip fucntion
            headingText = h.get_text(separator=" ").strip()  
            # adding tot the list
            headings.append(headingText)
        #print(headings)
        
        # extracting the bold and emphasized text
        boldText = []
        for b in soup.find_all(["b", "strong"]): # O(n)
            boldTextContent = b.get_text(separator=" ").strip()  
            boldText.append(boldTextContent)
        # print(boldText)

        # combine EVERYTHING you extracted together
        if title:
            keyElements.append(title)
        keyElements.extend(headings)
        keyElements.extend(boldText)
        
        # join everything into single string
        keyText = " ".join(filter(None, keyElements)).strip() # O(n)
        
        result = {
            "text": text.strip(),  
            "keyText": keyText 
        }
        
        return result
    except Exception as e:
        print(f"Error extracting text from HTML: {e}")
        return {"text": "", "keyText": ""}


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

    # need to call the findall method and extract
    tokens = re.findall(r'\b\w+\b', text.lower()) # O(n)

    stemmed_tokens = []
    for token in tokens: # O(n)
        # call the porterstemmer to reduce the token to its root
        stemmed_token = stemmer.stem(token) # O(n)
        stemmed_tokens.append(stemmed_token)
    #print(stemmed_tokens)
    return stemmed_tokens

def build_inverted_index(doc_id, tokens, inverted_index):
    '''
    Here, you need to build the inverted index with the token occurances from a given document

    doc_id = "/DEV/evoke_ics_uci_edu/page1.json"
    tokens = ['hello', 'world', 'hello']
    inverted_index = 
        {
        'hello': {'doc1.json': 2},
        'world': {'doc1.json': 1}
        }

    Time Complexity:
    - O(n): counting the term frequencies where n is the num of tokens
    - O(1): inserting the inverted index (individual)
    - Total: O(n + 1) = big O(n)
    '''
    # need to count how many times each word appears in doc id
    term_frequency = defaultdict(int)
    for token in tokens: # O(n)
        term_frequency[token] += 1 # O(1)
    for token, tf in term_frequency.items(): # O(n)
        inverted_index[token][doc_id] = tf # O(1)


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
    # save the file as partial_index...
    outputFileName = f"partial_index_{part_num}.json"
    
    indexDict = {}
    for token, postings in inverted_index.items(): # O(n)
        # converting the default dict to the regular dictionary
        indexDict[token] = dict(postings)
    
    # creating the dictionary to the JSON file
    with open(outputFileName, "w", encoding="utf-8") as f: # O(n)
        json.dump(indexDict, f, indent=4)
    
    # confirmation message so you know where the partial index is saved
    print(f"\nPartial index saved to {outputFileName}\n")
    return outputFileName

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
    invertedIndex = defaultdict(lambda: defaultdict(int))
    numOfDocuments = 0
    part_num = 1
    partialFiles = []

    for root, _, files in os.walk(directory): # O(n)
        # going through the subdirectories and files in the DEV folder
        for file in files: # O(n)
            if file.endswith(".json"):
                filePath = os.path.join(root, file)
                #print(filePath)
                htmlContent = read_json_file(filePath)
                #print(htmlContent)
                if not htmlContent:
                    continue
                # call all the methods to get the inverted index
                extractedText = extract_text_from_html(htmlContent)
                tokens = tokenize_and_normalize(extractedText["text"])
                build_inverted_index(filePath, tokens, invertedIndex)
                # incremement the number of documents each time
                numOfDocuments += 1

                if numOfDocuments % 500 == 0:
                    print(f"Processed {numOfDocuments} documents..")

                if numOfDocuments % 10000 == 0:
                    partialFilename = save_partial_index(invertedIndex, part_num)
                    partialFiles.append(partialFilename)
                    part_num += 1
                    # need to free memory for the current inverted index
                    invertedIndex.clear()

    if invertedIndex:
        partialFilename = save_partial_index(invertedIndex, part_num)
        # appending final filename to the list
        partialFiles.append(partialFilename)

    print(f"\nTotal Processed Documents: {numOfDocuments}")
    return partialFiles, numOfDocuments

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
    # need to initialize the empty dict
    finalIndex = defaultdict(lambda: defaultdict(int))

    # looping through every partial index file
    for partial in partial_files: # O(n)
        # opening and reading it through the open method
        with open(partial, "r", encoding="utf-8") as f:
            # loading the contents into the partialIndex
            partialIndex = json.load(f)
            # merging all the contents into the final inverted index
            for token, postings in partialIndex.items():  # O(n)
                for doc_id, tf in postings.items():  # O(n) -> these two add each other so total is O(n + n) = O(n)
                    # get the total term frequency in the final index
                    finalIndex[token][doc_id] += tf

    # writing the final merged inverted index in the json file
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump({token: dict(postings) for token, postings in finalIndex.items()}, f, indent=4)
    print(f"Inverted index saved to {output_filename}")


def generate_report(filename, num_docs):
    '''
    This is the generate report that has the summary of the num of documents that were indexed, 
    the count of the uqnique tokens, and the size of the index file.
    '''
    try:
        with open(filename, "r", encoding="utf-8") as f:
            invertedIndex = json.load(f)
        
        uniqueTokens = len(invertedIndex)
        # calculating it in kilobytes
        fileSize = os.path.getsize(filename) / 1024
        # printing all of the statements in the milestone 1 report
        print("\nMilestone 1 Report")
        print(f"Total Documents Indexed: {num_docs}")
        print(f"Unique Tokens: {uniqueTokens}")
        print(f"Index File Size: {fileSize:.2f} KB")

        # writing the report in the report.txt so it is easier to view the results
        with open("report.txt", "w") as f:
            f.write("Milestone 1 Report\n")
            f.write(f"Total Documents Indexed: {num_docs}\n")
            f.write(f"Unique Tokens: {uniqueTokens}\n")
            f.write(f"Index File Size: {fileSize:.2f} KB\n")
    
    except Exception as e:
        print(f"Error generating report: {e}")

if __name__ == "__main__":

    # Testing read_json_file:
    sampleFile = "/Users/ashritakuppili/Desktop/Inf121/project3/DEV/evoke_ics_uci_edu/0a50cb9b6d351654e0af8cbc6d20a6baa593e8c2bf1bef5b5b8d0e88fd2c4977.json"
    htmlContent = read_json_file(sampleFile)
    # print(htmlContent)

    extractedText = extract_text_from_html(htmlContent)
    # print(extractedText)

    projectDirectory = "/Users/ashritakuppili/Desktop/Inf121/project3/DEV"
    print("\nProccesing the documents in the dataset directory: ")
    
    partialFiles, numOfDocuments = process_all_documents(projectDirectory)
    # print(partialFiles)
    # print(numOfDocuments)
    merge_partial_indexes(partialFiles)
    generate_report("final_inverted_index.json", numOfDocuments)
    print("\nIndex process completed")