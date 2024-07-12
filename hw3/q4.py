import requests
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

# Load the model
model_name = 'multi-qa-distilbert-cos-v1'
embedding_model = SentenceTransformer(model_name)

# User question from Q1
user_question = "I just discovered the course. Can I still join it?"
v = embedding_model.encode(user_question)

# Load the documents with IDs
base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/documents-with-ids.json'
docs_url = f'{base_url}/{relative_url}?raw=1'
docs_response = requests.get(docs_url)
documents = docs_response.json()

# Filter the documents for "machine-learning-zoomcamp"
filtered_documents = [doc for doc in documents if doc['course'] == 'machine-learning-zoomcamp']

# Create embeddings for question and answer fields
embeddings = []

for doc in filtered_documents:
    question = doc.get('question', '')
    answer = doc.get('answer', '')
    qa_text = f'{question} {answer}'
    embedding = embedding_model.encode(qa_text)
    embeddings.append(embedding)

# Convert the list of embeddings to a numpy array
X = np.array(embeddings)

# VectorSearchEngine class definition
class VectorSearchEngine:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [self.documents[i] for i in idx]

# Initialize the search engine with the documents and embeddings
search_engine = VectorSearchEngine(documents=filtered_documents, embeddings=X)

# Load the ground truth dataset
base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
relative_url = '03-vector-search/eval/ground-truth-data.csv'
ground_truth_url = f'{base_url}/{relative_url}?raw=1'

df_ground_truth = pd.read_csv(ground_truth_url)
df_ground_truth = df_ground_truth[df_ground_truth.course == 'machine-learning-zoomcamp']
ground_truth = df_ground_truth.to_dict(orient='records')

# Inspect the keys of the ground truth data
print(f"Ground truth keys: {ground_truth[0].keys()}")

# Function to calculate hitrate
def calculate_hitrate(search_engine, ground_truth, num_results=5):
    hits = 0
    for gt in ground_truth:
        question = gt['question']
        true_document = gt['document']
        v_query = embedding_model.encode(question)
        results = search_engine.search(v_query, num_results=num_results)
        retrieved_documents = [res['question'] + " " + res.get('answer', '') for res in results]
        if true_document in retrieved_documents:
            hits += 1
    return hits / len(ground_truth)

# Calculate the hitrate of the VectorSearchEngine with num_results=5
hitrate = calculate_hitrate(search_engine, ground_truth, num_results=5)
print(f'Hitrate: {hitrate}')