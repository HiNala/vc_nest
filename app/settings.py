import os
from pathlib import Path

# Environment variables with Docker-friendly defaults
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")
EMBED_DIM = int(os.getenv("EMBED_DIM", "1536"))
CACHE_PATH = os.getenv("CACHE_PATH", "/app/data/embeddings_cache.npy")

# Ensure data directory exists
os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)

# Global variables (loaded on startup)
MOVIES = []
EMBEDS = None

def load_globals():
    """Load movies and embeddings into global variables."""
    global MOVIES, EMBEDS
    from app.data_loader import load_movies
    MOVIES, EMBEDS = load_movies()
    
def get_movies():
    """Get the global movies list."""
    return MOVIES
    
def get_embeddings():
    """Get the global embeddings matrix."""
    return EMBEDS 