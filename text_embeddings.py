from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

es = Elasticsearch('http://localhost:9200/')

index = 'test'

# Load your data 
data = []

# create elasticsearch index and mapping through it

if not es.indices.exists(index=index):
    es_index = {
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "embedding": {"type": "dense_vector", "dims": 768}
            }
        }
    }
    es.indices.create(index=index, body=es_index, ignore=[400])

model = SentenceTransformer('model_name')

for doc in data:
    embeds = model.encode(text, show_progress_bar=True)
    document = {
        "text": doc["text"],
        "embedding": embeds.tolist()
    }

    es.index(index=index, body=document)