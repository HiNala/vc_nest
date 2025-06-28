import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pinecone import Pinecone
import uvicorn

# ————— Models —————
class QueryRequest(BaseModel):
    query: str

class MovieHit(BaseModel):
    title: str
    score: float

class SearchResponse(BaseModel):
    movies: list[MovieHit]

# ————— Init Pinecone & FastAPI —————
app = FastAPI()

# Make sure you have your Pinecone key in the env var PINECONE_API_KEY
pc = Pinecone(api_key='pcsk_3R7ydo_J5XoRYLTZpVreUhC6UuUbjPfB258sBdqFX8VKNi9LnCJCgeugyPbwezXB6my4wP')
index = pc.Index("movies2")  # your existing index name
NAMESPACE = "example-namespace"

@app.post("/search", response_model=SearchResponse)
async def search(req: QueryRequest):
    q = req.query.strip()
    if not q:
        raise HTTPException(400, "Query must not be empty.")

    reranked_results = index.search(
        namespace="example-namespace",
        query={
            "top_k": 5,
            "inputs": {
                'text': req.query
            }
        },
        fields = "text"   
    )

    # ————— 3) Format results —————
    movies = [
        MovieHit(
            title=match.metadata.get("title", "<unknown>"),
            score=match.score
        )
        for match in reranked_results.matches
    ]
    return SearchResponse(movies=movies)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
