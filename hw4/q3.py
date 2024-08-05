import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Load the dataset from GitHub
github_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/data/results-gpt4o-mini.csv'
url = f'{github_url}?raw=1'
df = pd.read_csv(url)

# Use only the first 300 documents
df = df.iloc[:300]

# Load the embeddings model
model_name = 'multi-qa-mpnet-base-dot-v1'
embedding_model = SentenceTransformer(model_name)

# Function to normalize a vector
def normalize_vector(v):
    norm = np.sqrt((v * v).sum())
    return v / norm

# Function to compute cosine similarity between normalized embeddings of two answers
def compute_cosine_similarity(answer1, answer2):
    embedding1 = embedding_model.encode(answer1)
    embedding2 = embedding_model.encode(answer2)
    embedding1_norm = normalize_vector(embedding1)
    embedding2_norm = normalize_vector(embedding2)
    return np.dot(embedding1_norm, embedding2_norm)

# List to store the results (scores)
evaluations = []

# Compute cosine similarity for each answer pair and store the result
for index, row in df.iterrows():
    answer_llm = row['answer_llm']
    answer_orig = row['answer_orig']
    score = compute_cosine_similarity(answer_llm, answer_orig)
    evaluations.append(score)

# Calculate the 75th percentile of the cosine similarity scores
percentile_75 = np.percentile(evaluations, 75)
print(percentile_75)