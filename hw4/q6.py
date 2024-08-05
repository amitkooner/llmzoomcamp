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

# List to store ROUGE-2 F1 scores
rouge_2_f_scores = []

# Compute the ROUGE scores for each record and extract the ROUGE-2 F1 score
for index, row in df.iterrows():
    scores = rouge_scorer.get_scores(row['answer_llm'], row['answer_orig'])[0]
    rouge_2_f_scores.append(scores['rouge-2']['f'])

# Calculate the average ROUGE-2 F1 score
average_rouge_2_f_score = sum(rouge_2_f_scores) / len(rouge_2_f_scores)
print(average_rouge_2_f_score)