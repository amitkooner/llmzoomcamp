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

# Function to compute dot product between embeddings of two answers
def compute_dot_product(answer1, answer2):
    embedding1 = embedding_model.encode(answer1)
    embedding2 = embedding_model.encode(answer2)
    return np.dot(embedding1, embedding2)

# List to store the results (scores)
evaluations = []

# Compute dot product for each answer pair and store the result
for index, row in df.iterrows():
    answer_llm = row['answer_llm']
    answer_orig = row['answer_orig']
    score = compute_dot_product(answer_llm, answer_orig)
    evaluations.append(score)

# Calculate the 75th percentile of the scores
percentile_75 = np.percentile(evaluations, 75)
print(percentile_75)