import pandas as pd
from sentence_transformers import SentenceTransformer
from rouge import Rouge

# Load the dataset from GitHub
github_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/data/results-gpt4o-mini.csv'
url = f'{github_url}?raw=1'
df = pd.read_csv(url)

# Use only the first 300 documents
df = df.iloc[:300]

# Initialize the ROUGE scorer
rouge_scorer = Rouge()

# Select the answers at index 10
r = df.iloc[10]

# Compute the ROUGE scores
scores = rouge_scorer.get_scores(r['answer_llm'], r['answer_orig'])[0]

# Extract the F1 score for ROUGE-1
rouge_1_f_score = scores['rouge-1']['f']
print(rouge_1_f_score)