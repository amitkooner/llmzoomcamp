import pandas as pd
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

# Extract the F1 scores for ROUGE-1, ROUGE-2, and ROUGE-L
rouge_1_f_score = scores['rouge-1']['f']
rouge_2_f_score = scores['rouge-2']['f']
rouge_l_f_score = scores['rouge-l']['f']

# Compute the average F1 score
average_f_score = (rouge_1_f_score + rouge_2_f_score + rouge_l_f_score) / 3
print(average_f_score)