import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

# Initialize Elasticsearch client
es = Elasticsearch([{'host': 'localhost', 'port': 9200, 'scheme': 'http'}])

# Create the index with the appropriate settings
index_name = 'document_embeddings'

# Delete the index if it exists
if es.indices.exists(index=index_name):
    es.indices.delete(index=index_name)

# Create the index with the settings
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "question": {"type": "text"},
            "answer": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 768
            }
        }
    }
}

es.indices.create(index=index_name, body=index_settings)

# Load the documents with IDs
base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/documents-with-ids.json'
docs_url = f'{base_url}/{relative_url}?raw=1'
docs_response = requests.get(docs_url)
documents = docs_response.json()

# Filter the documents for "machine-learning-zoomcamp"
filtered_documents = [doc for doc in documents if doc['course'] == 'machine-learning-zoomcamp']

# Load the model
model_name = 'multi-qa-distilbert-cos-v1'
embedding_model = SentenceTransformer(model_name)

# Create embeddings for question and answer fields
embeddings = []

for doc in filtered_documents:
    question = doc.get('question', '')
    answer = doc.get('answer', '')
    qa_text = f'{question} {answer}'
    embedding = embedding_model.encode(qa_text)
    embeddings.append(embedding)

# Index the documents with embeddings
def generate_actions(documents, embeddings):
    for doc, embedding in zip(documents, embeddings):
        yield {
            "_index": index_name,
            "_source": {
                "question": doc['question'],
                "answer": doc.get('answer', ''),
                "embedding": embedding
            }
        }

actions = list(generate_actions(filtered_documents, embeddings))
bulk(es, actions)

# Load the ground truth dataset
ground_truth_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/03-vector-search/eval/ground-truth-data.csv?raw=1'
df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == 'machine-learning-zoomcamp']
ground_truth = df_ground_truth.to_dict(orient='records')

# Function to calculate hitrate
def calculate_hitrate_es(ground_truth, num_results=5):
    hits = 0
    for gt in ground_truth:
        question = gt['question']
        true_document = gt['document']
        v_query = embedding_model.encode(question)
        # Perform the search
        query = {
            "size": num_results,
            "query": {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                        "params": {"query_vector": v_query}
                    }
                }
            }
        }
        response = es.search(index=index_name, body=query)
        retrieved_documents = [hit['_source']['question'] + " " + hit['_source'].get('answer', '') for hit in response['hits']['hits']]
        if true_document in retrieved_documents:
            hits += 1
    return hits / len(ground_truth)

# Calculate the hitrate of Elasticsearch with num_results=5
hitrate_es = calculate_hitrate_es(ground_truth, num_results=5)
print(f'Hitrate: {hitrate_es}')