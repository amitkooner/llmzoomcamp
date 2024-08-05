import pandas as pd

# Load the dataset from GitHub
github_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/04-monitoring/data/results-gpt4o-mini.csv'
url = f'{github_url}?raw=1'
df = pd.read_csv(url)

# Print the column names to inspect them
print(df.columns)