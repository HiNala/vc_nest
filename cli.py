#!/usr/bin/env python3
"""
CLI for Movie Debate System

Usage examples:
    python cli.py --profiles demo_profiles.json
    python cli.py --profiles custom.json --verbose
    python cli.py --profiles profiles.json --max-rounds 3
"""

import click
import json
import asyncio
import os
import sys
from pathlib import Path

# Add the current directory to Python path for imports
sys.path.insert(0, str(Path(__file__).parent))

from app.settings import load_globals, OPENAI_API_KEY
from app.agents import run_debate
from app.utils import validate_profiles

@click.command()
@click.option('--profiles', '-p', 
              help='JSON file containing user profiles',
              type=click.Path(exists=True),
              required=True)
@click.option('--verbose', '-v', 
              is_flag=True,
              help='Enable verbose output')
@click.option('--max-rounds', '-r',
              default=5,
              help='Maximum number of debate rounds',
              type=int)
def main(profiles, verbose, max_rounds):
    """
    🎬 Movie Debate CLI - AI lawyers debate to find the perfect group movie!
    
    This CLI runs a complete movie recommendation debate where AI lawyers
    represent different user preferences and argue until reaching consensus.
    """
    
    # Setup
    if verbose:
        click.echo("🎬 Movie Debate System CLI")
        click.echo("=" * 50)
    
    # Check for OpenAI API key
    if not OPENAI_API_KEY:
        click.echo("❌ Error: OPENAI_API_KEY environment variable not set", err=True)
        click.echo("Please set your OpenAI API key:", err=True)
        click.echo("  export OPENAI_API_KEY=sk-your-key-here", err=True)
        sys.exit(1)
    
    # Load profiles
    try:
        with open(profiles, 'r') as f:
            data = json.load(f)
        
        # Handle different JSON formats
        if 'users' in data:
            user_profiles = data['users']
        elif isinstance(data, list):
            user_profiles = data
        else:
            user_profiles = [data]
            
        if verbose:
            click.echo(f"📁 Loaded {len(user_profiles)} user profiles from {profiles}")
            
    except Exception as e:
        click.echo(f"❌ Error loading profiles: {e}", err=True)
        sys.exit(1)
    
    # Validate profiles
    if not validate_profiles(user_profiles):
        click.echo("❌ Error: Invalid user profiles. Each profile must have 'name' and 'preferences' fields.", err=True)
        sys.exit(1)
    
    if len(user_profiles) < 2:
        click.echo("❌ Error: At least 2 user profiles required for a debate.", err=True)
        sys.exit(1)
    
    if verbose:
        click.echo("✓ Profiles validated successfully")
        for i, profile in enumerate(user_profiles, 1):
            click.echo(f"  {i}. {profile['name']}: {profile['preferences'][:50]}...")
    
    # Load movie data
    try:
        if verbose:
            click.echo("🔄 Loading movie database and embeddings...")
        load_globals()
        if verbose:
            click.echo("✓ Movie data loaded successfully")
    except Exception as e:
        click.echo(f"❌ Error loading movie data: {e}", err=True)
        click.echo("Make sure movies.csv exists and OpenAI API key is valid.", err=True)
        sys.exit(1)
    
    # Run the debate
    try:
        if verbose:
            click.echo("\n🎯 Starting movie debate...")
            click.echo("-" * 50)
        
        # Run async debate function
        result = asyncio.run(run_debate(user_profiles))
        
        # Display results
        winner = result['winner']
        participants = result['participants']
        rounds = result['total_rounds']
        
        click.echo("\n" + "=" * 60)
        click.echo("🏆 DEBATE RESULTS")
        click.echo("=" * 60)
        click.echo(f"🎬 Winning Movie: {winner['title']}")
        click.echo(f"📝 Description: {winner.get('overview', 'No description available')}")
        click.echo(f"👥 Participants: {', '.join(participants)}")
        click.echo(f"🥊 Total Rounds: {rounds}")
        
        if verbose and result.get('eliminated'):
            click.echo(f"\n💀 Eliminated Movies:")
            for i, movie in enumerate(result['eliminated'][-5:], 1):  # Show last 5
                click.echo(f"  {i}. {movie['title']}")
        
        click.echo("\n🎉 Recommendation complete! Enjoy your movie night! 🍿")
        
        # Export result option
        if click.confirm("\n💾 Save results to file?"):
            output_file = f"debate_result_{result['session_id'][:8]}.json"
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            click.echo(f"✓ Results saved to {output_file}")
        
    except KeyboardInterrupt:
        click.echo("\n⚠️  Debate interrupted by user", err=True)
        sys.exit(1)
    except Exception as e:
        click.echo(f"\n❌ Debate failed: {e}", err=True)
        if verbose:
            import traceback
            click.echo(traceback.format_exc(), err=True)
        sys.exit(1)

@click.command()
@click.option('--query', '-q',
              help='Search query for movies',
              required=True)
@click.option('--limit', '-l',
              default=5,
              help='Number of results to return',
              type=int)
def search(query, limit):
    """Search for movies using natural language."""
    try:
        load_globals()
        from app.mcp_tools import search_movies_by_text
        
        click.echo(f"🔍 Searching for: {query}")
        results = search_movies_by_text(query, limit)
        
        if not results:
            click.echo("No movies found matching your query.")
            return
        
        click.echo(f"\n📽️  Found {len(results)} movies:")
        click.echo("-" * 50)
        
        for i, movie in enumerate(results, 1):
            similarity = movie.get('similarity', 0)
            click.echo(f"{i}. {movie['title']} (similarity: {similarity:.3f})")
            click.echo(f"   {movie.get('overview', 'No description')[:100]}...")
            click.echo()
            
    except Exception as e:
        click.echo(f"❌ Search failed: {e}", err=True)

@click.group()
def cli():
    """🎬 Movie Debate System - AI-powered group movie recommendations"""
    pass

# Add commands to the group
cli.add_command(main, name='debate')
cli.add_command(search, name='search')

if __name__ == '__main__':
    # If called directly, run the main debate command
    if len(sys.argv) == 1:
        # No arguments, show help
        main(['--help'])
    else:
        # Check if first argument is a command
        if sys.argv[1] in ['debate', 'search']:
            cli()
        else:
            # Assume it's the main debate command
            main() 