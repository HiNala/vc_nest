import os
import numpy as np
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from app.settings import EMBED_DIM, CACHE_PATH, EMBED_MODEL, OPENAI_API_KEY

def load_movies():
    """Load movies and generate/cache embeddings."""
    # Look for CSV in multiple locations (local dev vs Docker)
    csv_paths = ["movies.csv", "/app/movies.csv", "./data/movies.csv", "data/movies.csv"]
    csv_path = next((p for p in csv_paths if os.path.exists(p)), None)
    
    if not csv_path:
        raise FileNotFoundError("movies.csv not found in expected locations")
    
    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} movies from {csv_path}")
    
    # Use cached embeddings if available
    if os.path.exists(CACHE_PATH):
        embeds = np.load(CACHE_PATH)
        print(f"✓ Loaded {len(embeds)} cached embeddings from {CACHE_PATH}")
    else:
        print("Generating embeddings...")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
            
        client = OpenAI(api_key=OPENAI_API_KEY)
        texts = [row["overview"] or row["title"] for _, row in df.iterrows()]
        
        embeds = []
        for text in tqdm(texts, desc="Generating embeddings"):
            response = client.embeddings.create(input=text, model=EMBED_MODEL)
            embeds.append(response.data[0].embedding)
        
        embeds = np.array(embeds, dtype="float32")
        
        # Cache for future use
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.save(CACHE_PATH, embeds)
        print(f"✓ Generated and cached {len(embeds)} embeddings to {CACHE_PATH}")
    
    # Normalize for cosine similarity via dot product
    embeds = embeds / np.linalg.norm(embeds, axis=1, keepdims=True)
    
    return df.to_dict("records"), embeds 