import openai
import os

# Set up your OpenAI API key from the environment variable
openai.api_key = os.getenv('OPENAI_API_KEY')

# Set the API base URL to your local server
openai.api_base = "http://localhost:11437/v1"

# Test prompt to get the completion
try:
    response = openai.ChatCompletion.create(
        model="gemma:2b",
        messages=[
            {"role": "user", "content": "What's the formula for energy?"}
        ],
        temperature=0.0
    )

    # Print the response to check the number of tokens
    print(response)

    # Extract the number of completion tokens
    completion_tokens = response['usage']['completion_tokens']
    print(f"Number of completion tokens: {completion_tokens}")

except openai.error.InvalidRequestError as e:
    print(f"Error: {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")