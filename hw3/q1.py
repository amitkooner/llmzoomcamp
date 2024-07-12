from sentence_transformers import SentenceTransformer

# Load the model
model_name = 'multi-qa-distilbert-cos-v1'
embedding_model = SentenceTransformer(model_name)

# User question
user_question = "I just discovered the course. Can I still join it?"

# Create the embedding
embedding = embedding_model.encode(user_question)

# Print the first value of the resulting vector
print(embedding[0])