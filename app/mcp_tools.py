# from fastapi_mcp import tool  # TODO: Fix MCP integration
import numpy as np
from openai import OpenAI
from app.settings import get_movies, get_embeddings, OPENAI_API_KEY

# @tool(name="vector_search", 
#       description="Find movies similar to query using semantic search")
def vector_search(query_embedding: list[float], k: int = 5) -> list[dict]:
    """Return top-k most similar movies."""
    movies = get_movies()
    embeds = get_embeddings()
    
    if not movies or embeds is None:
        return []
    
    q = np.asarray(query_embedding, dtype="float32")
    q = q / (np.linalg.norm(q) + 1e-9)  # Normalize query
    
    # Cosine similarity via dot product (pre-normalized vectors)
    similarities = embeds @ q
    top_indices = similarities.argsort()[-k:][::-1]
    
    return [
        {
            "index": int(idx),
            "title": movies[idx]["title"],
            "overview": movies[idx]["overview"],
            "similarity": float(similarities[idx])
        }
        for idx in top_indices
    ]

# @tool(name="search_movies_by_text",
#       description="Search for movies using natural language query")
def search_movies_by_text(query: str, k: int = 5) -> list[dict]:
    """Search movies using text query that gets embedded."""
    if not OPENAI_API_KEY:
        return []
        
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    # Generate embedding for the query
    response = client.embeddings.create(
        input=query,
        model="text-embedding-3-small"
    )
    query_embedding = response.data[0].embedding
    
    # Use vector search
    return vector_search(query_embedding, k)

# @tool(name="openai_chat",
#       description="Chat with OpenAI GPT model for generating arguments and responses")
def openai_chat(messages: list[dict], model: str = "gpt-4o-mini") -> str:
    """Generate chat completion using OpenAI."""
    if not OPENAI_API_KEY:
        return "OpenAI API key not available"
        
    client = OpenAI(api_key=OPENAI_API_KEY)
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error: {str(e)}" 