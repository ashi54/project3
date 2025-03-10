from flask import Blueprint, render_template, request
from Crawler_logic import Bool_Search, print_top_5_urls, merge_partial_indexes, process_all_documents

views = Blueprint('views', __name__)

# Process documents and build the final index
project_directory = "/Users/ashritakuppili/Desktop/Inf121/project3/crawler_web/DEV" #change to directory where files are stored at
partial_files = process_all_documents(project_directory)
final_index = merge_partial_indexes(partial_files)

@views.route('/', methods=['GET', 'POST'])
def home():
    '''
    This function handles the user search queries through a web interface.
    '''
    query = ''
    top_urls = []

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        
        if query:
            doc_results = Bool_Search(query, final_index)
            if doc_results:
                top_urls = print_top_5_urls(doc_results, final_index, query) or []

    return render_template('base.html', query=query, top_urls=top_urls)
