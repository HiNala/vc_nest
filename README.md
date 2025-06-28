# 🎬 Movie Debate System

An AI-powered group movie recommendation system where AI "lawyers" represent different user preferences and debate to find the perfect movie for your group.

## ✨ Features

- **AI Lawyer Agents**: Each user gets an AI lawyer powered by OpenAI GPT-4
- **Vector Search**: Semantic movie search using OpenAI embeddings
- **Tournament Debates**: Elimination-style arguments until consensus
- **Multiple Interfaces**: CLI, Web UI (Streamlit), and REST API
- **MCP Protocol**: Model Context Protocol for tool integration
- **Docker Ready**: Complete containerization with Docker Compose

## 🏗️ Architecture

```
👥 User Profiles → 🤖 AI Lawyers → 🔍 Vector Search → 🎬 Movie Database
                      ↓
                  🥊 Debate Tournament → 🏆 Winner
```

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key

### 1. Setup Environment
```bash
# Clone and navigate to the project
git clone <your-repo-url>
cd movie-debate-system

# Create environment file
cp .env.example .env
# Edit .env and add your OpenAI API key
```

### 2. Start the System
```bash
# Start all services
docker compose up --build

# Generate embeddings (first time only)
docker compose exec api python scripts/cache_embeddings.py
```

### 3. Access the Application
- **Web UI**: http://localhost:8501
- **API**: http://localhost:8080
- **API Docs**: http://localhost:8080/docs

### 4. Quick CLI Test
```bash
# Run a sample debate
docker compose run --rm cli python cli.py --profiles demo_profiles.json --verbose
```

## 📁 Project Structure

```
movie-debate-system/
├── app/                          # Core application
│   ├── main.py                   # FastAPI server + MCP
│   ├── settings.py               # Configuration
│   ├── data_loader.py            # CSV → embeddings pipeline
│   ├── mcp_tools.py              # MCP-exposed vector search
│   ├── agents.py                 # LangChain debate agents
│   └── utils.py                  # Scoring & pairing logic
├── streamlit_app/                # Web UI
│   ├── MovieDebate.py           # Streamlit application
│   └── Dockerfile               # UI container
├── scripts/
│   └── cache_embeddings.py      # Pre-compute embeddings
├── cli.py                        # Command-line interface
├── movies.csv                    # Movie dataset
├── demo_profiles.json            # Sample user preferences
├── docker-compose.yml            # Multi-service orchestration
├── Dockerfile                    # Main app container
└── requirements.txt              # Python dependencies
```

## 🎯 How It Works

1. **User Input**: Define 2-4 user profiles with movie preferences
2. **Lawyer Assignment**: Each user gets an AI lawyer agent
3. **Candidate Search**: Lawyers use vector search to find matching movies
4. **Tournament Debate**: Elimination rounds with AI arguments
5. **Consensus**: Final movie recommendation emerges

## 🔧 Usage Examples

### CLI Interface
```bash
# Basic debate
python cli.py --profiles demo_profiles.json

# With verbose output
python cli.py --profiles custom.json --verbose

# Search movies
python cli.py search --query "romantic comedy" --limit 5
```

### REST API
```bash
# Start a debate
curl -X POST http://localhost:8080/start-debate \
  -H "Content-Type: application/json" \
  -d '{
    "profiles": [
      {"name": "Alice", "preferences": "Romantic comedies", "dislikes": "Horror"},
      {"name": "Bob", "preferences": "Sci-fi thrillers", "dislikes": "Romance"}
    ]
  }'

# Check status
curl http://localhost:8080/debate-status/{session_id}

# Get result
curl http://localhost:8080/debate-result/{session_id}
```

### Sample Profile Format
```json
{
  "users": [
    {
      "name": "Alice",
      "preferences": "Love romantic comedies, feel-good movies with happy endings",
      "dislikes": "Horror, violence, sad endings"
    },
    {
      "name": "Bob",
      "preferences": "Sci-fi enthusiast, complex plots, technological themes",
      "dislikes": "Romance, musicals"
    }
  ]
}
```

## 🐳 Docker Services

### Available Services
```bash
# Start API only
docker compose up api

# Start API + Web UI
docker compose up api ui

# Run CLI
docker compose run --rm cli python cli.py --help

# View logs
docker compose logs -f api
```

### Service Endpoints
- **API**: http://localhost:8080
- **Streamlit UI**: http://localhost:8501
- **Health Check**: http://localhost:8080/health

## ⚙️ Configuration

### Environment Variables (.env)
```bash
# Required
OPENAI_API_KEY=sk-your-openai-api-key-here

# Optional
EMBED_MODEL=text-embedding-3-small
EMBED_DIM=1536
CACHE_PATH=/app/data/embeddings_cache.npy
```

### Customization
- **Movie Dataset**: Replace `movies.csv` with your own movies
- **Embedding Model**: Change `EMBED_MODEL` in `.env`
- **Debate Rules**: Modify `app/agents.py` for different tournament formats
- **Scoring Logic**: Update `app/utils.py` for custom argument scoring

## 🛠️ Development

### Local Development
```bash
# Setup virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Generate embeddings
python scripts/cache_embeddings.py

# Start API
uvicorn app.main:app --reload

# Start Streamlit UI
streamlit run streamlit_app/MovieDebate.py
```

### Adding New Movies
1. Add movies to `movies.csv`
2. Regenerate embeddings: `python scripts/cache_embeddings.py`
3. Restart services

### Extending Functionality
- **New Tools**: Add MCP tools in `app/mcp_tools.py`
- **Agent Behavior**: Modify prompts in `app/agents.py`
- **UI Features**: Enhance `streamlit_app/MovieDebate.py`

## 📊 Performance

### Expected Performance
- **Dataset Size**: Optimized for 1K-10K movies
- **Response Time**: 30-60 seconds per debate
- **Memory Usage**: ~100MB for 5K movies
- **Concurrent Users**: 10-20 simultaneous debates

### Optimization Tips
- Pre-compute embeddings with caching
- Use float32 for 2x memory savings
- Limit debate rounds to prevent long waits
- Scale with multiple API instances

## 🔍 Troubleshooting

### Common Issues
| Issue | Solution |
|-------|----------|
| `Cannot connect to API server` | Check `docker compose ps`, ensure API is healthy |
| `OpenAI API key not found` | Verify `.env` file exists with valid `OPENAI_API_KEY` |
| `Embeddings not generating` | Run `docker compose exec api python scripts/cache_embeddings.py` |
| `Port already in use` | Change ports in `docker-compose.yml` |
| `Out of memory` | Increase Docker memory limit or reduce dataset size |

### Debug Commands
```bash
# Check service status
docker compose ps

# View logs
docker compose logs api
docker compose logs ui

# Execute commands in container
docker compose exec api python scripts/cache_embeddings.py
docker compose exec api python cli.py --help

# Restart services
docker compose restart api
```

## 🚀 Deployment

### Production Deployment
```bash
# Build production images
docker compose build

# Deploy to cloud provider
# (Instructions vary by provider: AWS, GCP, Azure, etc.)

# Environment variables for production
export OPENAI_API_KEY=your-prod-key
export EMBED_MODEL=text-embedding-3-small
```

### Scaling
- Use Redis for session storage instead of in-memory
- Deploy multiple API instances behind a load balancer
- Use external vector database (Pinecone, Weaviate) for large datasets
- Add WebSocket support for real-time debate streaming

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make changes and test
4. Submit a pull request

### Adding New Features
- Follow existing code patterns
- Add tests for new functionality
- Update documentation
- Ensure Docker compatibility

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

- **OpenAI**: GPT-4 and embedding models
- **LangChain**: Agent framework
- **FastAPI**: High-performance web framework
- **Streamlit**: Rapid UI development
- **MCP Protocol**: Tool integration standard

---

## 🎬 Example Debate Flow

```
🎬 MOVIE DEBATE SESSION: abc123
============================================================
👥 Created 4 lawyers representing:
   - Alice: Love romantic comedies, feel-good movies...
   - Bob: Sci-fi enthusiast, complex plots...
   - Charlie: Action movies, superhero films...
   - Dana: Psychological thrillers, mystery...

🔍 CANDIDATE SELECTION PHASE
----------------------------------------
🎬 Alice's lawyer found 3 candidates
✓ Alice's top choice: The Princess Bride
🎬 Bob's lawyer found 3 candidates
✓ Bob's top choice: Inception
🎬 Charlie's lawyer found 3 candidates
✓ Charlie's top choice: Avengers: Endgame
🎬 Dana's lawyer found 3 candidates
✓ Dana's top choice: The Silence of the Lambs

🏁 TOURNAMENT PHASE: 4 movies enter the arena
--------------------------------------------------

🥊 ROUND 1
Remaining candidates: ['The Princess Bride', 'Inception', 'Avengers: Endgame', 'The Silence of the Lambs']

--- Match 1 ---
⚡ DEBATE: The Princess Bride vs Inception
🎯 Alice's lawyer argues for The Princess Bride...
   💬 The Princess Bride offers the perfect blend of romance, adventure, and humor that Alice craves...
🎯 Bob's lawyer argues for Inception...
   💬 Inception provides the complex, mind-bending narrative and technological themes that Bob loves...
🔥 Alice's lawyer rebuts...
   💬 While Inception is intellectually stimulating, The Princess Bride delivers emotional satisfaction...
🔥 Bob's lawyer rebuts...
   💬 The Princess Bride, though charming, lacks the sophisticated plot structure Bob appreciates...
🏆 Winner: The Princess Bride (representing Alice)

--- Match 2 ---
⚡ DEBATE: Avengers: Endgame vs The Silence of the Lambs
🏆 Winner: Avengers: Endgame (representing Charlie)

🥊 ROUND 2
--- Final Match ---
⚡ DEBATE: The Princess Bride vs Avengers: Endgame
🏆 Winner: The Princess Bride (representing Alice)

============================================================
🏆 FINAL RESULT
============================================================
**The Princess Bride**
A bedridden boy's grandfather reads him the story of a farmboy-turned-pirate who encounters numerous obstacles, enemies and allies in his quest to be reunited with his true love.

🎉 Recommendation for the group: The Princess Bride
============================================================
```

Perfect for movie nights! 🍿 