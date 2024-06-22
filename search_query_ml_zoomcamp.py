from elasticsearch import Elasticsearch

# Initialize Elasticsearch client
es = Elasticsearch(
    ['https://localhost:9200'],
    basic_auth=('elastic', 'XqqHEf4B_3-7EGAhnO0L'),
    verify_certs=False  # Only for testing purposes
)

index_name = "faq_index"

# Define the search query with a filter for the course field
search_query = {
    "query": {
        "bool": {
            "must": [
                {
                    "multi_match": {
                        "query": "How do I execute a command in a running docker container?",
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                }
            ],
            "filter": [
                {
                    "term": {
                        "course": "machine-learning-zoomcamp"
                    }
                }
            ]
        }
    },
    "size": 3
}

# Execute the search query
response = es.search(index=index_name, body=search_query)

# Print the top 3 results
hits = response['hits']['hits']
for i, hit in enumerate(hits):
    print(f"Result {i + 1} - Score: {hit['_score']}")
    print(f"Question: {hit['_source']['question']}")
    print(f"Text: {hit['_source']['text']}\n")

# Print the 3rd question specifically
third_question = hits[2]['_source']['question']
print("The 3rd question returned by the search engine is:", third_question)