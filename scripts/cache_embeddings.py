#!/usr/bin/env python3
"""
Pre-compute and cache movie embeddings for faster application startup.

This script should be run once to generate embeddings for all movies
in the dataset. The embeddings are cached to a .npy file for fast loading.

Usage:
    python scripts/cache_embeddings.py
    
Environment variables required:
    OPENAI_API_KEY - Your OpenAI API key
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.data_loader import load_movies
from app.settings import OPENAI_API_KEY, CACHE_PATH

def main():
    """Generate and cache embeddings."""
    print("🎬 Movie Embeddings Cache Generator")
    print("=" * 50)
    
    # Check for API key
    if not OPENAI_API_KEY:
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key:")
        print("  export OPENAI_API_KEY=sk-your-key-here")
        return 1
    
    # Check if cache already exists
    if os.path.exists(CACHE_PATH):
        print(f"⚠️  Cache file already exists: {CACHE_PATH}")
        response = input("Regenerate embeddings? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Aborted - using existing cache")
            return 0
        
        # Remove existing cache
        os.remove(CACHE_PATH)
        print("🗑️  Removed existing cache file")
    
    try:
        print("🔄 Generating embeddings (this may take several minutes)...")
        print(f"📁 Cache will be saved to: {CACHE_PATH}")
        
        # This will trigger embedding generation and caching
        movies, embeddings = load_movies()
        
        print(f"✅ Successfully generated embeddings for {len(movies)} movies")
        print(f"💾 Cache saved to: {CACHE_PATH}")
        print(f"📊 Embedding dimensions: {embeddings.shape}")
        print(f"💽 Cache file size: {os.path.getsize(CACHE_PATH) / 1024 / 1024:.2f} MB")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error generating embeddings: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 