import streamlit as st
import requests
import json
import time
import os
from typing import List, Dict

# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="🍿 Movie Debate",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.movie-card {
    border: 2px solid #e0e0e0;
    border-radius: 10px;
    padding: 20px;
    margin: 10px 0;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
}
.winner-card {
    border: 3px solid #gold;
    background: linear-gradient(135deg, #ffd700 0%, #ff8c00 100%);
    color: black;
}
.profile-card {
    border: 1px solid #ddd;
    border-radius: 8px;
    padding: 15px;
    margin: 10px 0;
    background-color: #f8f9fa;
}
</style>
""", unsafe_allow_html=True)

def check_api_health():
    """Check if the API is healthy and accessible."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        if response.status_code == 200:
            health_data = response.json()
            return True, health_data
        else:
            return False, None
    except Exception as e:
        return False, str(e)

def start_debate(profiles: List[Dict]) -> str:
    """Start a new debate session."""
    try:
        response = requests.post(
            f"{API_URL}/start-debate",
            json={"profiles": profiles},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()["session_id"]
        else:
            st.error(f"Failed to start debate: {response.json()}")
            return None
    except Exception as e:
        st.error(f"Error starting debate: {e}")
        return None

def get_debate_status(session_id: str) -> Dict:
    """Get the status of a debate session."""
    try:
        response = requests.get(f"{API_URL}/debate-status/{session_id}", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"status": "error", "error": "Failed to get status"}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def search_movies(query: str, limit: int = 5) -> List[Dict]:
    """Search for movies using the API."""
    try:
        response = requests.get(
            f"{API_URL}/movies/search",
            params={"query": query, "limit": limit},
            timeout=10
        )
        if response.status_code == 200:
            return response.json()["results"]
        else:
            return []
    except Exception as e:
        st.error(f"Search failed: {e}")
        return []

# Main App
st.title("🎬 AI Lawyer Movie Debate")
st.markdown("*Let AI lawyers represent your preferences and debate to find the perfect movie for your group!*")

# Check API health
with st.spinner("Connecting to backend..."):
    is_healthy, health_data = check_api_health()

if not is_healthy:
    st.error("❌ Cannot connect to the Movie Debate API server")
    st.info(f"Make sure the API is running at {API_URL}")
    st.info("Run: `docker compose up --build` or `uvicorn app.main:app`")
    st.stop()

st.success(f"✅ Connected to API - {health_data.get('movies_loaded', 0)} movies loaded")

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = None
if "debate_result" not in st.session_state:
    st.session_state.debate_result = None
if "profiles" not in st.session_state:
    st.session_state.profiles = []

# Sidebar for configuration and info
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
    
    # Movie search tool
    st.header("🔍 Movie Search")
    search_query = st.text_input("Search movies:", placeholder="romantic comedy")
    if search_query:
        with st.spinner("Searching..."):
            search_results = search_movies(search_query, 3)
        
        if search_results:
            st.write("**Results:**")
            for movie in search_results:
                st.write(f"• {movie['title']} ({movie.get('similarity', 0):.2f})")
                with st.expander(f"About {movie['title']}"):
                    st.write(movie.get('overview', 'No description available'))

# Main content area
tab1, tab2, tab3 = st.tabs(["🎭 Create Debate", "📊 Results", "⚙️ Advanced"])

with tab1:
    st.header("👥 User Preferences")
    st.markdown("Add 2-4 people with their movie preferences:")
    
    # Profile input form
    with st.form("profile_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            name = st.text_input("Name", placeholder="Alice")
            preferences = st.text_area(
                "Movie Preferences", 
                placeholder="I love romantic comedies, feel-good movies with happy endings...",
                height=100
            )
        
        with col2:
            dislikes = st.text_input(
                "Dislikes (optional)", 
                placeholder="Horror, violence, sad endings..."
            )
            age_group = st.selectbox(
                "Age Group (for reference)",
                ["Not specified", "Kids", "Teens", "Adults", "Seniors"]
            )
        
        add_profile = st.form_submit_button("➕ Add Person", type="primary")
        
        if add_profile and name and preferences:
            profile = {
                "name": name.strip(),
                "preferences": preferences.strip(),
                "dislikes": dislikes.strip() if dislikes else ""
            }
            st.session_state.profiles.append(profile)
            st.success(f"Added {name} to the debate!")
            st.rerun()
    
    # Show current profiles
    if st.session_state.profiles:
        st.subheader("👥 Current Participants")
        
        for i, profile in enumerate(st.session_state.profiles):
            with st.container():
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"**{profile['name']}**")
                    st.write(f"Likes: {profile['preferences']}")
                    if profile.get('dislikes'):
                        st.write(f"Dislikes: {profile['dislikes']}")
                
                with col2:
                    if st.button("🗑️", key=f"remove_{i}"):
                        st.session_state.profiles.pop(i)
                        st.rerun()
        
        # Start debate button
        if len(st.session_state.profiles) >= 2:
            if st.button("🚀 Start Movie Debate!", type="primary", use_container_width=True):
                with st.spinner("🤖 AI lawyers are assembling..."):
                    session_id = start_debate(st.session_state.profiles)
                
                if session_id:
                    st.session_state.session_id = session_id
                    st.success("Debate started! Check the Results tab...")
                    time.sleep(1)
                    st.rerun()
        else:
            st.info("👆 Add at least 2 people to start a debate")
        
        # Clear all button
        if st.button("🧹 Clear All", type="secondary"):
            st.session_state.profiles = []
            st.rerun()
    
    else:
        st.info("➕ Add your first person above to get started")

with tab2:
    st.header("🏆 Debate Results")
    
    if st.session_state.session_id:
        # Auto-refresh while debate is running
        status_placeholder = st.empty()
        result_placeholder = st.empty()
        
        # Get current status
        status = get_debate_status(st.session_state.session_id)
        
        with status_placeholder.container():
            if status["status"] == "running":
                st.info("💭 Debate in progress... AI lawyers are arguing!")
                
                # Progress bar (fake but fun)
                progress_bar = st.progress(0)
                for i in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(i + 1)
                
                # Auto-refresh every 5 seconds
                time.sleep(5)
                st.rerun()
                
            elif status["status"] == "completed":
                st.success("🎉 Debate completed!")
                result = status.get("result", {})
                
                if result:
                    with result_placeholder.container():
                        # Winner announcement
                        winner = result.get("winner", {})
                        
                        st.markdown(f"""
                        <div class="movie-card winner-card">
                            <h2>🏆 WINNING MOVIE</h2>
                            <h3>{winner.get('title', 'Unknown Movie')}</h3>
                            <p>{winner.get('overview', 'No description available')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Debate stats
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("👥 Participants", len(result.get("participants", [])))
                        
                        with col2:
                            st.metric("🥊 Rounds", result.get("total_rounds", 0))
                        
                        with col3:
                            st.metric("💀 Movies Eliminated", len(result.get("eliminated", [])))
                        
                        # Participants
                        st.subheader("👥 Debate Participants")
                        participants = result.get("participants", [])
                        if participants:
                            st.write(", ".join(participants))
                        
                        # Eliminated movies
                        eliminated = result.get("eliminated", [])
                        if eliminated:
                            st.subheader("💀 Eliminated Movies")
                            for movie in eliminated[-5:]:  # Show last 5
                                st.write(f"• {movie.get('title', 'Unknown')}")
                        
                        # Action buttons
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if st.button("🔄 Start New Debate", type="primary"):
                                st.session_state.session_id = None
                                st.session_state.debate_result = None
                                st.rerun()
                        
                        with col2:
                            # Download results as JSON
                            result_json = json.dumps(result, indent=2)
                            st.download_button(
                                "💾 Download Results",
                                result_json,
                                file_name=f"debate_result_{result.get('session_id', 'unknown')[:8]}.json",
                                mime="application/json"
                            )
            
            elif status["status"] == "failed":
                st.error(f"❌ Debate failed: {status.get('error', 'Unknown error')}")
                
                if st.button("🔄 Try Again"):
                    st.session_state.session_id = None
                    st.rerun()
    
    else:
        st.info("👈 Start a debate in the Create Debate tab to see results here")

with tab3:
    st.header("⚙️ Advanced Options")
    
    # API status
    st.subheader("🔌 API Status")
    if health_data:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Movies Loaded", health_data.get("movies_loaded", 0))
        
        with col2:
            st.metric("Active Debates", health_data.get("active_debates", 0))
        
        with col3:
            st.metric("API Version", health_data.get("version", "Unknown"))
    
    # Quick actions
    st.subheader("🚀 Quick Actions")
    
    if st.button("🔍 Test Movie Search"):
        test_results = search_movies("action adventure", 3)
        if test_results:
            st.success("✅ Search working!")
            for movie in test_results:
                st.write(f"• {movie['title']}")
        else:
            st.error("❌ Search failed")
    
    if st.button("🏥 Check API Health"):
        is_healthy, health_data = check_api_health()
        if is_healthy:
            st.success("✅ API is healthy!")
            st.json(health_data)
        else:
            st.error("❌ API health check failed")
    
    # Load sample profiles
    st.subheader("📝 Sample Profiles")
    
    sample_profiles = [
        {
            "name": "Alice",
            "preferences": "Romantic comedies, feel-good movies with happy endings, strong character development",
            "dislikes": "Horror, excessive violence, sad endings"
        },
        {
            "name": "Bob", 
            "preferences": "Sci-fi thrillers, complex plots, technological themes, mind-bending concepts",
            "dislikes": "Romance, musicals, slow pacing"
        },
        {
            "name": "Charlie",
            "preferences": "Action movies, superhero films, adventure stories, fast-paced entertainment",
            "dislikes": "Slow burns, art house films, excessive dialogue"
        },
        {
            "name": "Dana",
            "preferences": "Psychological thrillers, mystery with plot twists, dark themes",
            "dislikes": "Predictable plots, children's movies, overly happy endings"
        }
    ]
    
    if st.button("📋 Load Sample Profiles"):
        st.session_state.profiles = sample_profiles
        st.success("✅ Loaded 4 sample profiles!")
        st.rerun()
    
    # Session management
    st.subheader("🗃️ Session Management")
    
    if st.session_state.session_id:
        st.info(f"Current session: {st.session_state.session_id}")
        
        if st.button("🗑️ Clear Session"):
            st.session_state.session_id = None
            st.session_state.debate_result = None
            st.success("Session cleared!")
    else:
        st.info("No active session")

# Footer
st.markdown("---")
st.markdown("🍿 **Movie Debate System** - Powered by AI lawyers and vector search")
st.markdown(f"API Status: Connected to {API_URL}") 