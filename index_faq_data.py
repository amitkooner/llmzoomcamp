import requests 
import json
from elasticsearch import Elasticsearch
import urllib3

# Suppress InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Fetch the documents data from the URL
docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']
    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

# Initialize Elasticsearch client with basic_auth
es = Elasticsearch(
    ['https://localhost:9200'],
    basic_auth=('elastic', 'XqqHEf4B_3-7EGAhnO0L'),
    verify_certs=False  # Only for testing purposes
)

index_name = "faq_index"

# Define the mappings
mappings = {
    "mappings": {
        "properties": {
            "course": {"type": "keyword"},
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"}
        }
    }
}

# Create the index if it doesn't exist
if not es.indices.exists(index=index_name):
    es.indices.create(index=index_name, body=mappings)

# Index the documents
for i, doc in enumerate(documents):
    es.index(index=index_name, id=i, body=doc)

print("Data indexed successfully")