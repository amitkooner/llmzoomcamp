from elasticsearch import Elasticsearch

# Initialize Elasticsearch client
es = Elasticsearch(
    ['https://localhost:9200'],
    basic_auth=('elastic', 'XqqHEf4B_3-7EGAhnO0L'),
    verify_certs=False  # Only for testing purposes
)

index_name = "faq_index"

# Define the search query
search_query = {
    "query": {
        "multi_match": {
            "query": "How do I execute a command in a running docker container?",
            "fields": ["question^4", "text"],
            "type": "best_fields"
        }
    }
}

# Execute the search query
response = es.search(index=index_name, body=search_query)

# Print the top ranking result
top_hit = response['hits']['hits'][0]
print("Top ranking result score:", top_hit['_score'])
print("Top ranking result source:", top_hit['_source'])