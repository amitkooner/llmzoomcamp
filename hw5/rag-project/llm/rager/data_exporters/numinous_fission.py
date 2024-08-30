import json
from typing import Dict, List, Union
from datetime import datetime

import numpy as np
from elasticsearch import Elasticsearch
from mage_ai.data_preparation.variable_manager import set_global_variable


@data_exporter
def elasticsearch(documents: List[Dict[str, Union[Dict, List[int], str]]], *args, **kwargs):
    connection_string = kwargs.get('connection_string', 'http://localhost:9200')
    
    # Adjusting the index name with a time-based prefix
    index_name_prefix = kwargs.get('index_name', 'documents')
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    index_name = f"{index_name_prefix}_{current_time}"
    print("index name:", index_name)
    
    # Saving the index name as a global variable
    set_global_variable('RAGic', 'index_name', index_name)  # Pipeline name is RAGic
    
    number_of_shards = kwargs.get('number_of_shards', 1)
    number_of_replicas = kwargs.get('number_of_replicas', 0)
    dimensions = kwargs.get('dimensions')

    if dimensions is None and len(documents) > 0:
        document = documents[0]
        dimensions = len(document.get('embedding') or [])

    es_client = Elasticsearch(connection_string)

    print(f'Connecting to Elasticsearch at {connection_string}')

    # Adjusting index settings
    index_settings = {
        "settings": {
            "number_of_shards": number_of_shards,
            "number_of_replicas": number_of_replicas,
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"},
                "document_id": {"type": "keyword"}
            }
        }
    }

    # Recreate the index by deleting if it exists and then creating with new settings
    if es_client.indices.exists(index=index_name):
        es_client.indices.delete(index=index_name)
        print(f'Index {index_name} deleted')

    es_client.indices.create(index=index_name, body=index_settings)
    print('Index created with properties:')
    print(json.dumps(index_settings, indent=2))
    print('Embedding dimensions:', dimensions)

    count = len(documents)
    print(f'Indexing {count} documents to Elasticsearch index {index_name}')
    
    last_document = None
    for idx, document in enumerate(documents):
        if idx % 100 == 0:
            print(f'{idx + 1}/{count}')

        # Removing the embedding handling since we're not using vector search
        # if isinstance(document['embedding'], np.ndarray):
        #     document['embedding'] = document['embedding'].tolist()

        es_client.index(index=index_name, document=document)
        last_document = document

    # Only print the last document if one was processed
    if last_document:
        print(last_document)
    else:
        print("No documents were processed.")

    return [[d['embedding'] for d in documents[:10]]]