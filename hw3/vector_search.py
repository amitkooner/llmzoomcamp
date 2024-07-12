import requests
import numpy as np
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

# Perform a search with the query vector
results = search_engine.search(v, num_results=5)

# Print the results
for result in results:
    print(result)