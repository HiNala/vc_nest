# Now create store it in a pinecone index
from pinecone import Pinecone
import time
import json
import random
import itertools
import os
import pandas as pd

def chunks(iterable, batch_size=200):
    """A helper function to break an iterable into chunks of size batch_size."""
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))


def create_vector_database( batch_size: int = 95, index_name: str = "movies2", namespace: str = "example-namespace"):
# Initialize a Pinecone client with your API key
    pc = Pinecone(api_key='pcsk_3R7ydo_J5XoRYLTZpVreUhC6UuUbjPfB258sBdqFX8VKNi9LnCJCgeugyPbwezXB6my4wP')
    data_movies = pd.read_json('movies.json')

    # Create a dense index with integrated embedding (or connect if exists)
    if not pc.has_index(index_name):
        print(f"Creating index '{index_name}' with integrated embeddings for 'llama-text-embed-v2'...")
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model":"llama-text-embed-v2",
                "field_map":{"text": "text"} # Pinecone will look for 'chunk_text' in your records
            }
        )
        print(f"Index '{index_name}' created. It might take a moment to be ready.")
        time.sleep(10) # Give a moment for the index to initialize
    else:
        print(f"Index '{index_name}' already exists. Connecting to it.")

    dense_index = pc.Index(index_name)

    # Upsert the records into a namespace
    print(f"Upserting {len(data_movies)} records into namespace {index_name} in batches...")

    # Define batch size, matching your chunks function or Pinecone limits (e.g., 100-200 for upsert_records)
    total_upserted_count = 0

    for record_chunk in chunks(data_movies, batch_size=batch_size):
        try:
            print(f"Upserting batch of {len(record_chunk)} records...")
            dense_index.upsert_records(namespace="example-namespace", records=list(record_chunk))
            total_upserted_count += len(record_chunk)
            print(f"Successfully upserted batch. Total upserted so far: {total_upserted_count}")
        except Exception as e:
            print(f"Error upserting batch: {e}")
            # Optionally, add more sophisticated error handling here, like retries or logging failed batches

    print(f"Successfully upserted {total_upserted_count} records in total to namespace 'example-namespace'.")



create_vector_database()