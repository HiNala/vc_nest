import asyncio
import uuid
from typing import List, Dict, Optional
from openai import OpenAI
from app.settings import OPENAI_API_KEY
from app.mcp_tools import search_movies_by_text, openai_chat
from app.utils import (
    create_tournament_pairs, determine_winner, format_movie_info,
    create_debate_context, delay_for_drama, validate_profiles, normalize_preferences
)

class MovieLawyer:
    """AI agent representing one user's movie preferences."""
    
    def __init__(self, user_profile: Dict, session_id: str):
        self.profile = normalize_preferences(user_profile)
        self.session_id = session_id
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.candidate_movies = []
    
    async def find_candidate_movies(self, k: int = 5) -> List[Dict]:
        """Find movies that match this user's preferences."""
        if not self.client:
            return []
        
        # Create search query from preferences
        query = f"{self.profile['preferences']}"
        if self.profile.get('dislikes'):
            query += f" but not {self.profile['dislikes']}"
        
        # Use the MCP tool to search for movies
        movies = search_movies_by_text(query, k)
        self.candidate_movies = movies
        
        print(f"🎬 {self.profile['name']}'s lawyer found {len(movies)} candidates")
        return movies
    
    async def argue_for_movie(self, movie: Dict, opponent_movie: Dict, context: str) -> str:
        """Generate argument for why this movie is better than the opponent's."""
        if not self.client:
            return f"I recommend {movie['title']} for {self.profile['name']}."
        
        messages = [
            {
                "role": "system",
                "content": f"""You are a skilled lawyer representing {self.profile['name']} in a movie recommendation debate.
                
Your client's preferences: {self.profile['preferences']}
Your client's dislikes: {self.profile.get('dislikes', 'None specified')}

You must argue why your movie choice is superior to the opponent's choice for your client.
Be persuasive, specific, and reference the movie details. Keep arguments to 2-3 sentences max."""
            },
            {
                "role": "user",
                "content": f"""Context: {context}

Your movie: {movie['title']} - {movie.get('overview', 'No description')}
Opponent's movie: {opponent_movie['title']} - {opponent_movie.get('overview', 'No description')}

Argue why your movie is the better choice for {self.profile['name']}."""
            }
        ]
        
        return openai_chat(messages, "gpt-4o-mini")
    
    async def rebut_argument(self, opponent_argument: str, my_movie: Dict, opponent_movie: Dict) -> str:
        """Generate a rebuttal to the opponent's argument."""
        if not self.client:
            return f"I still believe {my_movie['title']} is the better choice."
        
        messages = [
            {
                "role": "system", 
                "content": f"""You are a lawyer defending your movie recommendation for {self.profile['name']}.
                
Your client's preferences: {self.profile['preferences']}
Your client's dislikes: {self.profile.get('dislikes', 'None specified')}

Counter the opponent's argument with a strong rebuttal. Be concise but convincing."""
            },
            {
                "role": "user",
                "content": f"""Your movie: {my_movie['title']}
Opponent's movie: {opponent_movie['title']}
Opponent's argument: {opponent_argument}

Provide a strong rebuttal defending your choice."""
            }
        ]
        
        return openai_chat(messages, "gpt-4o-mini")

async def conduct_movie_debate(movie1: Dict, movie2: Dict, lawyer1: MovieLawyer, 
                              lawyer2: MovieLawyer, context: str) -> Dict:
    """Conduct a debate between two movies with their respective lawyers."""
    print(f"⚡ DEBATE: {movie1['title']} vs {movie2['title']}")
    
    # Opening arguments
    print(f"🎯 {lawyer1.profile['name']}'s lawyer argues for {movie1['title']}...")
    arg1 = await lawyer1.argue_for_movie(movie1, movie2, context)
    print(f"   💬 {arg1}")
    
    await delay_for_drama(0.5)
    
    print(f"🎯 {lawyer2.profile['name']}'s lawyer argues for {movie2['title']}...")
    arg2 = await lawyer2.argue_for_movie(movie2, movie1, context)
    print(f"   💬 {arg2}")
    
    await delay_for_drama(0.5)
    
    # Rebuttals
    print(f"🔥 {lawyer1.profile['name']}'s lawyer rebuts...")
    rebut1 = await lawyer1.rebut_argument(arg2, movie1, movie2)
    print(f"   💬 {rebut1}")
    
    await delay_for_drama(0.5)
    
    print(f"🔥 {lawyer2.profile['name']}'s lawyer rebuts...")
    rebut2 = await lawyer2.rebut_argument(arg1, movie2, movie1)
    print(f"   💬 {rebut2}")
    
    # Determine winner based on all arguments
    all_profiles = [lawyer1.profile, lawyer2.profile]
    combined_arg1 = f"{arg1} {rebut1}"
    combined_arg2 = f"{arg2} {rebut2}"
    
    winner = determine_winner(movie1, movie2, combined_arg1, combined_arg2, all_profiles)
    
    winner_name = lawyer1.profile['name'] if winner == movie1 else lawyer2.profile['name']
    print(f"🏆 Winner: {winner['title']} (representing {winner_name})")
    
    return winner

async def run_debate(profiles: List[Dict], session_id: Optional[str] = None) -> Dict:
    """Execute the complete lawyer debate tournament."""
    if session_id is None:
        session_id = uuid.uuid4().hex
    
    print(f"\n🎬 MOVIE DEBATE SESSION: {session_id}")
    print("=" * 60)
    
    # Validate profiles
    if not validate_profiles(profiles):
        raise ValueError("Invalid user profiles provided")
    
    if len(profiles) < 2:
        raise ValueError("At least 2 user profiles required for debate")
    
    # Create lawyers for each user
    lawyers = [MovieLawyer(profile, session_id) for profile in profiles]
    print(f"👥 Created {len(lawyers)} lawyers representing:")
    for lawyer in lawyers:
        print(f"   - {lawyer.profile['name']}: {lawyer.profile['preferences']}")
    
    print("\n🔍 CANDIDATE SELECTION PHASE")
    print("-" * 40)
    
    # Each lawyer finds their top movie candidate
    candidates = []
    for lawyer in lawyers:
        movies = await lawyer.find_candidate_movies(3)  # Get top 3, use best one
        if movies:
            candidates.append(movies[0])  # Take the top choice
            print(f"✓ {lawyer.profile['name']}'s top choice: {movies[0]['title']}")
        else:
            print(f"❌ {lawyer.profile['name']}'s lawyer found no suitable movies")
    
    if len(candidates) < 2:
        raise ValueError("Not enough movie candidates found for debate")
    
    print(f"\n🏁 TOURNAMENT PHASE: {len(candidates)} movies enter the arena")
    print("-" * 50)
    
    eliminated = []
    round_num = 1
    max_rounds = 5
    
    # Tournament elimination rounds
    while len(candidates) > 1 and round_num <= max_rounds:
        print(f"\n🥊 ROUND {round_num}")
        print(f"Remaining candidates: {[m['title'] for m in candidates]}")
        
        context = create_debate_context(round_num, max_rounds, eliminated)
        pairs = create_tournament_pairs(candidates)
        winners = []
        
        for i, (movie1, movie2) in enumerate(pairs):
            print(f"\n--- Match {i+1} ---")
            
            # Find lawyers representing these movies
            lawyer1 = next((l for l in lawyers if movie1 in l.candidate_movies), lawyers[0])
            lawyer2 = next((l for l in lawyers if movie2 in l.candidate_movies), lawyers[1])
            
            winner = await conduct_movie_debate(movie1, movie2, lawyer1, lawyer2, context)
            winners.append(winner)
            
            # Track eliminated movie
            loser = movie2 if winner == movie1 else movie1
            eliminated.append(loser)
            print(f"💀 Eliminated: {loser['title']}")
        
        # Handle odd number (bye)
        if len(candidates) % 2 == 1:
            bye_movie = candidates[-1]
            winners.append(bye_movie)
            print(f"🎯 {bye_movie['title']} advances automatically (bye)")
        
        candidates = winners
        round_num += 1
        
        await delay_for_drama(1.0)
    
    # Final result
    final_winner = candidates[0] if candidates else eliminated[-1]
    
    print("\n" + "=" * 60)
    print("🏆 FINAL RESULT")
    print("=" * 60)
    print(format_movie_info(final_winner))
    print(f"\n🎉 Recommendation for the group: {final_winner['title']}")
    print("=" * 60)
    
    return {
        "session_id": session_id,
        "winner": final_winner,
        "eliminated": eliminated,
        "total_rounds": round_num - 1,
        "participants": [lawyer.profile['name'] for lawyer in lawyers]
    } 