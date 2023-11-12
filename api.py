from fastapi import FastAPI
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import uvicorn

es = Elasticsearch()

app = FastAPI(
    title='Load data',
    description='Load Embeddings vectors to Elasticsearch',
    version='0.1'
)

@app.get('/search')
async def search(query: str):
    model = SentenceTransformer('quora-distilbert-multilingual')
    embedding = model.encode(query, show_progress_bar=False)

    # Build the Elasticsearch script query
    script_query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": embedding.tolist()}
            }
        }
    }

    # Execute the search query
    search_results = es.search(index="test1", body={"query": script_query})

    results = search_results["hits"]["hits"]
    return {"results": results}

if __name__ == '__main__':
    uvicorn(app, '0.0.0.0', port=5984)
