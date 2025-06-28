from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
# from fastapi_mcp import attach_mcp  # TODO: Fix MCP integration
from pydantic import BaseModel
from typing import List, Dict, Optional
import uuid
import logging
import asyncio

from app.settings import load_globals, get_movies
from app.agents import run_debate
from app.utils import validate_profiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load data on startup
logger.info("Loading movies and embeddings...")
try:
    load_globals()
    logger.info(f"✓ Data loaded successfully - {len(get_movies())} movies available")
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    # Don't exit, allow the app to start for health checks

app = FastAPI(
    title="Movie Debate MCP Server", 
    description="AI lawyer debate system for group movie recommendations",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://ui:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Attach MCP tools
# attach_mcp(app)  # TODO: Fix MCP integration

# Pydantic models
class UserProfile(BaseModel):
    name: str
    preferences: str
    dislikes: Optional[str] = ""

class DebateRequest(BaseModel):
    profiles: List[UserProfile]

class DebateResponse(BaseModel):
    session_id: str
    status: str
    message: str

# Global storage for debate sessions (in production, use Redis or database)
debate_sessions: Dict[str, Dict] = {}

@app.post("/start-debate", response_model=DebateResponse)
async def start_debate(request: DebateRequest, tasks: BackgroundTasks):
    """Start a new movie debate session."""
    # Validate profiles
    profiles_dict = [profile.dict() for profile in request.profiles]
    
    if not validate_profiles(profiles_dict):
        raise HTTPException(status_code=400, detail="Invalid user profiles provided")
    
    if len(profiles_dict) < 2:
        raise HTTPException(status_code=400, detail="At least 2 user profiles required")
    
    if len(profiles_dict) > 8:
        raise HTTPException(status_code=400, detail="Maximum 8 user profiles allowed")
    
    session_id = uuid.uuid4().hex
    logger.info(f"Starting debate session {session_id} with {len(profiles_dict)} profiles")
    
    # Initialize session
    debate_sessions[session_id] = {
        "status": "running",
        "profiles": profiles_dict,
        "result": None,
        "error": None
    }
    
    # Run debate in background
    tasks.add_task(run_debate_background, profiles_dict, session_id)
    
    return DebateResponse(
        session_id=session_id,
        status="started",
        message=f"Debate started with {len(profiles_dict)} participants"
    )

async def run_debate_background(profiles: List[Dict], session_id: str):
    """Background task to run the debate."""
    try:
        logger.info(f"Running debate for session {session_id}")
        result = await run_debate(profiles, session_id)
        
        # Update session with result
        debate_sessions[session_id].update({
            "status": "completed",
            "result": result
        })
        
        logger.info(f"Debate completed for session {session_id}: {result['winner']['title']}")
        
    except Exception as e:
        logger.error(f"Debate failed for session {session_id}: {e}")
        debate_sessions[session_id].update({
            "status": "failed",
            "error": str(e)
        })

@app.get("/debate-status/{session_id}")
async def get_debate_status(session_id: str):
    """Get the status of a debate session."""
    if session_id not in debate_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return debate_sessions[session_id]

@app.get("/debate-result/{session_id}")
async def get_debate_result(session_id: str):
    """Get the final result of a completed debate."""
    if session_id not in debate_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = debate_sessions[session_id]
    
    if session["status"] == "running":
        raise HTTPException(status_code=202, detail="Debate still in progress")
    
    if session["status"] == "failed":
        raise HTTPException(status_code=500, detail=f"Debate failed: {session['error']}")
    
    return session["result"]

@app.delete("/debate-session/{session_id}")
async def delete_debate_session(session_id: str):
    """Delete a debate session."""
    if session_id not in debate_sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    del debate_sessions[session_id]
    return {"message": "Session deleted successfully"}

@app.get("/active-sessions")
async def get_active_sessions():
    """Get all active debate sessions."""
    return {
        "sessions": [
            {
                "session_id": sid,
                "status": session["status"],
                "participants": len(session["profiles"])
            }
            for sid, session in debate_sessions.items()
        ],
        "total": len(debate_sessions)
    }

@app.post("/quick-debate")
async def quick_debate(request: DebateRequest):
    """Run a synchronous debate (for testing/demo purposes)."""
    profiles_dict = [profile.dict() for profile in request.profiles]
    
    if not validate_profiles(profiles_dict):
        raise HTTPException(status_code=400, detail="Invalid user profiles provided")
    
    if len(profiles_dict) < 2:
        raise HTTPException(status_code=400, detail="At least 2 user profiles required")
    
    try:
        result = await run_debate(profiles_dict)
        return result
    except Exception as e:
        logger.error(f"Quick debate failed: {e}")
        raise HTTPException(status_code=500, detail=f"Debate failed: {str(e)}")

@app.get("/movies/search")
async def search_movies(query: str, limit: int = 10):
    """Search movies by text query."""
    from app.mcp_tools import search_movies_by_text
    
    try:
        movies = search_movies_by_text(query, limit)
        return {"query": query, "results": movies, "count": len(movies)}
    except Exception as e:
        logger.error(f"Movie search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/movies/stats")
async def get_movie_stats():
    """Get statistics about the movie database."""
    movies = get_movies()
    return {
        "total_movies": len(movies),
        "sample_movies": movies[:5] if movies else [],
        "embedding_status": "loaded" if movies else "not_loaded"
    }

@app.get("/health")
def health_check():
    """Health check endpoint for Docker."""
    movies = get_movies()
    return {
        "status": "healthy",
        "movies_loaded": len(movies),
        "active_debates": len(debate_sessions),
        "version": "1.0.0"
    }

@app.get("/")
def root():
    """Root endpoint with API info."""
    return {
        "message": "🎬 Movie Debate API",
        "description": "AI lawyer debate system for group movie recommendations",
        "docs": "/docs",
        "health": "/health",
        "version": "1.0.0",
        "endpoints": {
            "start_debate": "POST /start-debate",
            "get_status": "GET /debate-status/{session_id}",
            "get_result": "GET /debate-result/{session_id}",
            "search_movies": "GET /movies/search?query=...",
            "mcp_tools": "GET /mcp/tools"
        }
    }

# Cleanup old sessions periodically (simple implementation)
@app.on_event("startup")
async def startup_cleanup():
    """Clean up old sessions on startup."""
    global debate_sessions
    debate_sessions = {}
    logger.info("Application started, session storage cleared")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 