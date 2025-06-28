import random
import asyncio
from typing import List, Dict, Tuple

def create_tournament_pairs(candidates: List[Dict]) -> List[Tuple[Dict, Dict]]:
    """Create pairs for tournament-style elimination."""
    pairs = []
    shuffled = candidates.copy()
    random.shuffle(shuffled)
    
    for i in range(0, len(shuffled) - 1, 2):
        pairs.append((shuffled[i], shuffled[i + 1]))
    
    return pairs

def score_argument(argument: str, movie: Dict, user_profile: Dict) -> float:
    """Simple scoring function for arguments (can be enhanced with ML)."""
    # Basic scoring based on argument length and keyword matching
    base_score = min(len(argument.split()) / 50.0, 1.0)  # Normalize by word count
    
    # Bonus for mentioning user preferences
    prefs = user_profile.get("preferences", "").lower()
    if any(word in argument.lower() for word in prefs.split()):
        base_score += 0.3
    
    # Penalty for mentioning dislikes
    dislikes = user_profile.get("dislikes", "").lower()
    if any(word in argument.lower() for word in dislikes.split()):
        base_score -= 0.2
    
    return max(0.0, min(1.0, base_score))

def determine_winner(movie1: Dict, movie2: Dict, argument1: str, argument2: str, 
                    profiles: List[Dict]) -> Dict:
    """Determine the winner between two movies based on arguments and profiles."""
    score1 = sum(score_argument(argument1, movie1, profile) for profile in profiles)
    score2 = sum(score_argument(argument2, movie2, profile) for profile in profiles)
    
    # Add some randomness to prevent ties
    score1 += random.uniform(-0.1, 0.1)
    score2 += random.uniform(-0.1, 0.1)
    
    return movie1 if score1 > score2 else movie2

def format_movie_info(movie: Dict) -> str:
    """Format movie information for display."""
    return f"**{movie['title']}**\n{movie.get('overview', 'No description available')}"

def create_debate_context(round_num: int, total_rounds: int, eliminated: List[Dict]) -> str:
    """Create context string for the current debate round."""
    context = f"Round {round_num}/{total_rounds}"
    if eliminated:
        context += f" | Eliminated: {', '.join(m['title'] for m in eliminated[-3:])}"
    return context

async def delay_for_drama(seconds: float = 1.0):
    """Add dramatic pause between debate rounds."""
    await asyncio.sleep(seconds)

def validate_profiles(profiles: List[Dict]) -> bool:
    """Validate that user profiles have required fields."""
    required_fields = ["name", "preferences"]
    
    for profile in profiles:
        if not all(field in profile for field in required_fields):
            return False
        if not profile["name"].strip() or not profile["preferences"].strip():
            return False
    
    return True

def normalize_preferences(profile: Dict) -> Dict:
    """Normalize and clean user preferences."""
    cleaned = profile.copy()
    cleaned["preferences"] = profile.get("preferences", "").strip()
    cleaned["dislikes"] = profile.get("dislikes", "").strip()
    cleaned["name"] = profile.get("name", "").strip()
    
    return cleaned 