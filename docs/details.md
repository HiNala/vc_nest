# Movie Debate System - Implementation Guide

## Overview

This system creates a unique movie recommendation engine where 4 AI "lawyers" represent different user preferences and debate to find the best movie for a group. Each agent uses vector search to find movies matching their assigned user profile, then engages in elimination rounds until reaching consensus.

### Core Concept
- **4 User Profiles** → Each gets an AI "lawyer" agent
- **Vector Search** → Agents find movies using RAG with OpenAI embeddings  
- **Debate Rounds** → Agents argue for their recommendations in tournament-style elimination
- **Consensus Winner** → Final movie recommendation emerges from the debate process

### Why This Approach?
- Handles conflicting preferences in group settings
- Transparent reasoning through debate logs
- Scalable to different group sizes and preference types
- Combines semantic search with argumentative reasoning

---

## Technical Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Backend** | FastAPI + fastapi-mcp | REST API + MCP protocol server |
| **AI Agents** | LangChain + OpenAI GPT-4 | Lawyer agents for debate logic |
| **Vector Search** | OpenAI Embeddings + NumPy | Movie similarity matching |
| **Database** | CSV + NumPy cache | Simple dataset with fast retrieval |
| **Interface** | CLI (primary) + Streamlit (optional) | User interaction |
| **Runtime** | Python 3.12 | Modern Python features |

---

## Architecture

```
┌─ User Profiles (JSON) ─┐
│ 4 different preferences │
└─────────┬───────────────┘
          │
┌─────────▼───────────────┐
│   AI Lawyer Agents      │
│ (LangChain + OpenAI)    │
└─────────┬───────────────┘
          │
┌─────────▼───────────────┐
│   Vector Search Tool    │
│ (OpenAI Embeddings)     │
└─────────┬───────────────┘
          │
┌─────────▼───────────────┐
│   Movie Database        │
│ (CSV + NumPy cache)     │
└─────────────────────────┘
```

---

## Project Structure

```
movie-debate-system/
├── README.md
├── requirements.txt
├── movies.csv                    # Movie dataset
├── demo_profiles.json            # Sample user preferences
├── .env.example                  # Environment template
├── Dockerfile                    # Main app container
├── docker-compose.yml            # Multi-service orchestration
├── .dockerignore                 # Docker build exclusions
│
├── app/                          # Core application
│   ├── main.py                   # FastAPI server + MCP
│   ├── settings.py               # Configuration
│   ├── data_loader.py            # CSV → embeddings pipeline
│   ├── mcp_tools.py              # MCP-exposed vector search
│   ├── agents.py                 # LangChain debate agents
│   └── utils.py                  # Scoring & pairing logic
│
├── cli.py                        # Command-line interface
├── streamlit_app/                # Web UI service
│   ├── MovieDebate.py
│   └── Dockerfile                # Streamlit container
├── scripts/
│   └── cache_embeddings.py       # Pre-compute embeddings
└── data/                         # Persistent data volume
    └── embeddings_cache.npy      # Generated embeddings
```

---

## Docker Configuration

### Main Application Dockerfile
```dockerfile
# Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/
COPY cli.py .
COPY scripts/ ./scripts/
COPY movies.csv .
COPY demo_profiles.json .

# Create data directory for persistent storage
RUN mkdir -p /app/data

# Expose FastAPI port
EXPOSE 8000

# Default command runs the FastAPI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Streamlit UI Dockerfile
```dockerfile
# streamlit_app/Dockerfile
FROM python:3.12-slim

WORKDIR /app

# Install streamlit and requests
RUN pip install streamlit==1.35.0 requests==2.31.0 sseclient-py==1.8.0

# Copy streamlit app
COPY MovieDebate.py .

# Expose Streamlit port
EXPOSE 8501

# Run Streamlit
CMD ["streamlit", "run", "MovieDebate.py", "--server.address", "0.0.0.0"]
```

### Docker Compose Configuration
```yaml
# docker-compose.yml
version: '3.8'

services:
  api:
    build: .
    container_name: movie-debate-api
    ports:
      - "8000:8000"
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - EMBED_MODEL=${EMBED_MODEL:-text-embedding-3-small}
      - EMBED_DIM=${EMBED_DIM:-1536}
      - CACHE_PATH=/app/data/embeddings_cache.npy
    volumes:
      - ./data:/app/data
      - ./movies.csv:/app/movies.csv
      - ./demo_profiles.json:/app/demo_profiles.json
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    restart: unless-stopped

  ui:
    build: ./streamlit_app
    container_name: movie-debate-ui
    ports:
      - "8501:8501"
    depends_on:
      api:
        condition: service_healthy
    environment:
      - API_URL=http://api:8000
    restart: unless-stopped

  # CLI service for one-off commands
  cli:
    build: .
    container_name: movie-debate-cli
    profiles: ["cli"]  # Only start when explicitly requested
    environment:
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - EMBED_MODEL=${EMBED_MODEL:-text-embedding-3-small}
      - EMBED_DIM=${EMBED_DIM:-1536}
      - CACHE_PATH=/app/data/embeddings_cache.npy
    volumes:
      - ./data:/app/data
      - ./movies.csv:/app/movies.csv
      - ./demo_profiles.json:/app/demo_profiles.json
    command: ["python", "cli.py", "--profiles", "demo_profiles.json"]

volumes:
  movie_data:
    driver: local
```

### Docker Ignore File
```dockerignore
# .dockerignore
.git
.gitignore
README.md
.env
.venv
__pycache__
*.pyc
*.pyo
*.pyd
.Python
build
develop-eggs
dist
downloads
eggs
.eggs
lib
lib64
parts
sdist
var
wheels
*.egg-info/
.pytest_cache
.coverage
htmlcov
.tox
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis
.streamlit
```

---

## Quick Start with Docker

### 1. Environment Setup
```bash
# Clone repository
git clone <your-repo-url>
cd movie-debate-system

# Copy environment template
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Environment Variables (.env)
```bash
# .env
OPENAI_API_KEY=your_openai_api_key_here
EMBED_MODEL=text-embedding-3-small
EMBED_DIM=1536
```

### 3. Start All Services
```bash
# Build and start API + UI
docker compose up --build

# Or run in background
docker compose up -d --build
```

### 4. Access the Application
- **Web UI**: http://localhost:8501
- **API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 5. Generate Embeddings (First Run)
```bash
# Generate embeddings cache
docker compose exec api python scripts/cache_embeddings.py

# Or run as one-off container
docker compose run --rm api python scripts/cache_embeddings.py
```

---

## Docker Usage Examples

### CLI Commands
```bash
# Run CLI with default profiles
docker compose run --rm cli

# Run CLI with custom profiles
docker compose run --rm cli python cli.py --profiles your_profiles.json

# Run with verbose output
docker compose run --rm cli python cli.py --profiles demo_profiles.json --verbose
```

### Development Mode
```bash
# Start with live reload (for development)
docker compose up --build

# View logs
docker compose logs -f api
docker compose logs -f ui

# Restart specific service
docker compose restart api
```

### Production Deployment
```bash
# Production build
docker compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Scale API service
docker compose up -d --scale api=3
```

---

## Key Dependencies

```txt
# Core Framework
fastapi==0.110.0
fastapi-mcp==0.3.4
uvicorn[standard]==0.29.0

# AI & ML
langchain==0.1.15
langchain-openai==0.0.8
openai==1.28.1
numpy==1.26.4

# Data Processing
pandas==2.2.2
tqdm==4.66.2

# Interface
click==8.1.7          # CLI
streamlit==1.35.0     # Optional web UI
```

---

## Core Implementation

### Docker-Ready Settings Configuration
```python
# app/settings.py
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
```

### Updated Data Loading with Docker Paths
```python
# app/data_loader.py
import os, numpy as np, pandas as pd
from openai import OpenAI
from tqdm import tqdm
from app.settings import EMBED_DIM, CACHE_PATH, EMBED_MODEL

def load_movies():
    """Load movies and generate/cache embeddings."""
    # Look for CSV in multiple locations (local dev vs Docker)
    csv_paths = ["movies.csv", "/app/movies.csv", "./data/movies.csv"]
    csv_path = next((p for p in csv_paths if os.path.exists(p)), None)
    
    if not csv_path:
        raise FileNotFoundError("movies.csv not found in expected locations")
    
    df = pd.read_csv(csv_path)
    
    # Use cached embeddings if available
    if os.path.exists(CACHE_PATH):
        embeds = np.load(CACHE_PATH)
        print(f"✓ Loaded {len(embeds)} cached embeddings from {CACHE_PATH}")
    else:
        print("Generating embeddings...")
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        texts = [row["overview"] or row["title"] for _, row in df.iterrows()]
        
        embeds = np.vstack([
            client.embeddings.create(input=text, model=EMBED_MODEL).data[0].embedding
            for text in tqdm(texts)
        ]).astype("float32")
        
        # Cache for future use
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        np.save(CACHE_PATH, embeds)
        print(f"✓ Generated and cached {len(embeds)} embeddings to {CACHE_PATH}")
    
    # Normalize for cosine similarity via dot product
    embeds /= np.linalg.norm(embeds, axis=1, keepdims=True)
    return df.to_dict("records"), embeds
```

### Docker-Ready FastAPI Main
```python
# app/main.py
from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi_mcp import attach_mcp
from app.settings import load_globals
from app.agents import run_debate
import uuid
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data on startup
logger.info("Loading movies and embeddings...")
load_globals()
logger.info("✓ Data loaded successfully")

app = FastAPI(
    title="Movie Debate MCP Server", 
    description="AI lawyer debate system for group movie recommendations",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://ui:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach MCP tools
attach_mcp(app)

@app.post("/start-debate")
async def start_debate(profiles: list[dict], tasks: BackgroundTasks):
    """Start a new movie debate session."""
    session_id = uuid.uuid4().hex
    logger.info(f"Starting debate session {session_id} with {len(profiles)} profiles")
    
    # Run debate in background
    tasks.add_task(run_debate, profiles, session_id)
    
    return {"session_id": session_id, "status": "started"}

@app.get("/health")
def health_check():
    """Health check endpoint for Docker."""
    return {"status": "healthy", "movies_loaded": len(MOVIES) if MOVIES else 0}

@app.get("/")
def root():
    """Root endpoint with API info."""
    return {
        "message": "Movie Debate API",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### MCP Vector Search Tool
```python
# app/mcp_tools.py
from fastapi_mcp import tool
import numpy as np

@tool(name="vector_search", 
      description="Find movies similar to query using semantic search")
def vector_search(query_embedding: list[float], k: int = 5) -> list[dict]:
    """Return top-k most similar movies."""
    q = np.asarray(query_embedding, dtype="float32")
    q /= np.linalg.norm(q) + 1e-9
    
    # Cosine similarity via dot product (pre-normalized vectors)
    similarities = EMBEDS @ q
    top_indices = similarities.argsort()[-k:][::-1]
    
    return [
        {
            "index": int(idx),
            "title": MOVIES[idx]["title"],
            "overview": MOVIES[idx]["overview"],
            "similarity": float(similarities[idx])
        }
        for idx in top_indices
    ]
```

### Debate Agent System
```python
# app/agents.py
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent
from langchain.tools import Tool

class MovieLawyer:
    """AI agent representing one user's movie preferences."""
    
    def __init__(self, user_profile: dict):
        self.profile = user_profile
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
        self.tools = [
            Tool.from_function(
                func=self.search_movies,
                name="search_movies",
                description="Find movies matching user preferences"
            )
        ]
    
    async def argue_for_movie(self, movie: dict, context: str) -> str:
        """Generate argument for why this movie fits the user profile."""
        prompt = f"""
        You are a lawyer arguing for the movie "{movie['title']}" for this user:
        Profile: {self.profile['preferences']}
        
        Movie: {movie['overview']}
        Context: {context}
        
        Make a compelling 2-3 sentence argument for why this movie is perfect for your client.
        """
        response = await self.llm.agenerate([prompt])
        return response.generations[0][0].text

async def run_debate(profiles: list[dict]) -> dict:
    """Execute the lawyer debate tournament."""
    # Initialize lawyers
    lawyers = [MovieLawyer(profile) for profile in profiles]
    
    # Round 1: Each lawyer finds their top candidate
    candidates = []
    for lawyer in lawyers:
        movies = await lawyer.search_movies()
        candidates.append(movies[0])  # Top choice
    
    # Tournament elimination rounds
    while len(candidates) > 1:
        next_round = []
        for i in range(0, len(candidates), 2):
            if i + 1 < len(candidates):
                winner = await debate_pair(candidates[i], candidates[i+1], lawyers)
                next_round.append(winner)
            else:
                next_round.append(candidates[i])  # Bye
        candidates = next_round
    
    return candidates[0]  # Final winner
```

---

## Usage Examples

### CLI Usage
```bash
# Basic recommendation
python cli.py --profiles demo_profiles.json

# With debug output
python cli.py --profiles profiles.json --verbose

# Custom number of rounds
python cli.py --profiles profiles.json --max-rounds 3
```

### Sample Profile Format
```json
{
  "users": [
    {
      "name": "Alice",
      "preferences": "Love romantic comedies and feel-good movies with happy endings",
      "dislikes": "Horror, violence, sad endings"
    },
    {
      "name": "Bob", 
      "preferences": "Sci-fi enthusiast, loves complex plots and technological themes",
      "dislikes": "Romance, musicals"
    },
    {
      "name": "Charlie",
      "preferences": "Action movie fan, enjoys superhero films and adventure",
      "dislikes": "Slow pacing, art house films"
    },
    {
      "name": "Dana",
      "preferences": "Appreciates psychological thrillers and mystery with plot twists",
      "dislikes": "Predictable plots, children's movies"
    }
  ]
}
```

---

## API Endpoints

### Docker Services
- **API Server**: http://localhost:8000
- **Streamlit UI**: http://localhost:8501
- **API Documentation**: http://localhost:8000/docs

### Available Endpoints
```bash
# Health check
GET  /health
# Response: {"status": "healthy", "movies_loaded": 1234}

# Start debate
POST /start-debate
# Body: [{"name": "Alice", "preferences": "...", "dislikes": "..."}]
# Response: {"session_id": "abc123", "status": "started"}

# MCP tool schema
GET  /mcp/schema
# Response: MCP tool definitions

# API documentation
GET  /docs
# Interactive Swagger UI
```

### Example API Usage
```bash
# Health check
curl http://localhost:8000/health

# Start debate
curl -X POST http://localhost:8000/start-debate \
  -H "Content-Type: application/json" \
  -d '[
    {"name": "Alice", "preferences": "Romantic comedies", "dislikes": "Horror"},
    {"name": "Bob", "preferences": "Sci-fi thrillers", "dislikes": "Romance"}
  ]'
```

---

### Docker-Ready Streamlit App
```python
# streamlit_app/MovieDebate.py
import streamlit as st
import requests
import json
import os
from typing import List, Dict

# API configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="🍿 Movie Debate",
    page_icon="🎬",
    layout="wide"
)

st.title("🍿 AI Lawyer Movie Debate")
st.markdown("*Let AI lawyers debate to find the perfect movie for your group!*")

# Check API health
try:
    health = requests.get(f"{API_URL}/health", timeout=5).json()
    st.success(f"✓ Connected to API - {health['movies_loaded']} movies loaded")
except:
    st.error("❌ Cannot connect to API server")
    st.stop()

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "debate_result" not in st.session_state:
    st.session_state.debate_result = None

# User profile input
st.header("👥 User Preferences")

profiles = []
cols = st.columns(2)

for i in range(4):
    with cols[i % 2]:
        st.subheader(f"User {i+1}")
        name = st.text_input(f"Name", key=f"name_{i}", placeholder=f"Person {i+1}")
        prefs = st.text_area(
            f"Movie Preferences", 
            key=f"prefs_{i}",
            placeholder="What kinds of movies do you enjoy?",
            height=100
        )
        dislikes = st.text_input(
            f"Dislikes", 
            key=f"dislikes_{i}",
            placeholder="What do you want to avoid?"
        )
        
        if name and prefs:
            profiles.append({
                "name": name,
                "preferences": prefs,
                "dislikes": dislikes or "None specified"
            })

# Start debate
st.header("🎭 Movie Debate")

if len(profiles) >= 2:
    if st.button("🚀 Start Debate!", type="primary", use_container_width=True):
        with st.spinner("🤖 AI lawyers are debating..."):
            try:
                response = requests.post(
                    f"{API_URL}/start-debate",
                    json=profiles,
                    timeout=120
                )
                result = response.json()
                st.session_state.session_id = result["session_id"]
                st.success("Debate started! Check back in a moment...")
                st.rerun()
            except Exception as e:
                st.error(f"Error starting debate: {e}")

    # Show results if available
    if st.session_state.session_id:
        try:
            # In a real implementation, you'd poll for results
            # For demo purposes, show placeholder
            st.info("💭 Debate in progress... Results will appear here")
            
            # Mock result for demo
            if st.button("🎬 Show Mock Result"):
                st.session_state.debate_result = {
                    "title": "The Princess Bride",
                    "overview": "A classic fairy tale adventure with romance, comedy, and action",
                    "reasoning": "This movie satisfies multiple preferences: romance for Alice, adventure for Charlie, clever plot for Dana, and it's family-friendly enough to avoid Bob's violence concerns."
                }
                st.rerun()
                
        except Exception as e:
            st.error(f"Error checking debate status: {e}")

# Display results
if st.session_state.debate_result:
    st.header("🏆 Winning Movie")
    result = st.session_state.debate_result
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.image("https://via.placeholder.com/300x450/FF6B6B/FFFFFF?text=Movie+Poster", 
                caption=result["title"])
    
    with col2:
        st.subheader(result["title"])
        st.write(result["overview"])
        
        st.subheader("🎯 Why This Movie Won")
        st.write(result["reasoning"])
        
        if st.button("🔄 Start New Debate"):
            st.session_state.session_id = None
            st.session_state.debate_result = None
            st.rerun()

else:
    st.info("👆 Enter at least 2 user preferences to start a debate")

# Sidebar with instructions
with st.sidebar:
    st.header("📋 How It Works")
    st.markdown("""
    1. **Enter Preferences**: Add 2-4 users with their movie preferences
    2. **AI Lawyers**: Each user gets an AI lawyer to represent their tastes
    3. **Vector Search**: Lawyers find movies using semantic search
    4. **Debate Tournament**: Lawyers argue for their choices in elimination rounds
    5. **Consensus**: The final winner becomes your group recommendation
    """)
    
    st.header("💡 Tips")
    st.markdown("""
    - Be specific about preferences (genres, themes, moods)
    - Mention strong dislikes to avoid bad matches
    - Include variety in your group for interesting debates
    """)
```

---

## Performance & Scaling

### Optimization Tips
- **Embedding Cache**: Pre-compute all movie embeddings
- **Float32**: Use `float32` instead of `float64` for 2x memory savings
- **Batch Processing**: Generate embeddings in batches for large datasets
- **Timeout Guards**: Limit debate rounds to prevent infinite loops

### Expected Performance
- **Dataset Size**: Optimized for 1K-10K movies
- **Response Time**: 30-60 seconds per complete debate
- **Memory Usage**: ~100MB for 5K movies with embeddings
- **Concurrent Users**: 10-20 simultaneous debates

---

## Troubleshooting

### Docker Issues
| Issue | Solution |
|-------|----------|
| `Cannot connect to API server` | Check `docker compose ps`, ensure API is healthy |
| `Permission denied` errors | Run `chmod +x` on script files, check volume mounts |
| `Port already in use` | Change ports in `docker-compose.yml` or stop conflicting services |
| `OpenAI API key not found` | Verify `.env` file exists and `OPENAI_API_KEY` is set |
| `Embeddings not generating` | Run `docker compose exec api python scripts/cache_embeddings.py` |
| `Out of memory` | Increase Docker memory limit in Docker Desktop settings |

### Application Issues  
| Issue | Solution |
|-------|----------|
| OpenAI rate limits | Add delays, use smaller batches, cache embeddings |
| Memory errors | Use float32, reduce embedding dimensions, increase Docker memory |
| Long debate times | Set max_rounds=5, max_turns_per_pair=6 |
| Vector search slow | Ensure embeddings are L2-normalized |
| Streamlit connection issues | Check API_URL environment variable in UI container |

### Debugging Commands
```bash
# Check service status
docker compose ps

# View service logs
docker compose logs api
docker compose logs ui

# Execute commands in running containers
docker compose exec api python scripts/cache_embeddings.py
docker compose exec api python cli.py --help

# Check container environment
docker compose exec api env | grep OPENAI
docker compose exec api ls -la /app/data/

# Restart specific services
docker compose restart api
docker compose restart ui
```

---

## Future Enhancements

- **Database Integration**: Replace CSV with PostgreSQL + pgvector
- **Advanced Agents**: Add personality traits, debate strategies
- **Group Size Scaling**: Support 2-8 users instead of fixed 4
- **Real-time Streaming**: WebSocket-based live debate updates
- **Preference Learning**: Adapt to user feedback over time
- **Multi-modal**: Include movie posters, trailers in recommendations

---

## Quick Start Checklist

### Docker Setup (Recommended)
- [ ] Clone repository
- [ ] Install Docker & Docker Compose
- [ ] Copy `.env.example` to `.env`
- [ ] Add OpenAI API key to `.env`
- [ ] Run `docker compose up --build`
- [ ] Generate embeddings: `docker compose exec api python scripts/cache_embeddings.py`
- [ ] Access UI at http://localhost:8501
- [ ] Test CLI: `docker compose run --rm cli`

### Local Development (Alternative)
- [ ] Install Python 3.12+
- [ ] Create virtual environment
- [ ] Install requirements: `pip install -r requirements.txt`
- [ ] Set up `.env` file
- [ ] Generate embeddings: `python scripts/cache_embeddings.py`
- [ ] Start API: `uvicorn app.main:app --reload`
- [ ] Start UI: `streamlit run streamlit_app/MovieDebate.py`

**Time to first recommendation: ~3 minutes with Docker**

---

## Sample Files

### Environment Template (.env.example)
```bash
# Required: Your OpenAI API key
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional: Embedding model configuration  
EMBED_MODEL=text-embedding-3-small
EMBED_DIM=1536

# Docker paths (don't change unless you know what you're doing)
CACHE_PATH=/app/data/embeddings_cache.npy
```

### Sample User Profiles (demo_profiles.json)
```json
{
  "users": [
    {
      "name": "Alice",
      "preferences": "Love romantic comedies, feel-good movies with happy endings, strong character development",
      "dislikes": "Horror, excessive violence, sad endings, zombie movies"
    },
    {
      "name": "Bob", 
      "preferences": "Sci-fi enthusiast, complex plots, technological themes, mind-bending concepts",
      "dislikes": "Romance, musicals, slow pacing, romantic comedies"
    },
    {
      "name": "Charlie",
      "preferences": "Action movies, superhero films, adventure, fast-paced entertainment",
      "dislikes": "Slow burns, art house films, excessive dialogue, period dramas"
    },
    {
      "name": "Dana",
      "preferences": "Psychological thrillers, mystery with plot twists, dark themes, sophisticated narratives",
      "dislikes": "Predictable plots, children's movies, overly happy endings, simple stories"
    }
  ]
}
```