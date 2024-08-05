import pandas as pd
from sentence_transformers import SentenceTransformer

# Correct URL for the dataset
github_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/data/results-gpt4o-mini.csv'
url = f'{github_url}?raw=1'

# Load the dataset from GitHub
df = pd.read_csv(url)

# Use only the first 300 documents
df = df.iloc[:300]

# Load the embeddings model
model_name = 'multi-qa-mpnet-base-dot-v1'
embedding_model = SentenceTransformer(model_name)

# Get the first LLM answer
answer_llm = df.iloc[0]['answer_llm']

# Create the embeddings for the first LLM answer
embeddings = embedding_model.encode(answer_llm)

# Print the first value of the resulting vector
first_value = embeddings[0]
print(first_value)