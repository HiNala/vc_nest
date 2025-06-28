import json
import time
import itertools
import re
import unicodedata
from pinecone import Pinecone

def chunks(iterable, batch_size=95):
    it = iter(iterable)
    chunk = tuple(itertools.islice(it, batch_size))
    while chunk:
        yield chunk
        chunk = tuple(itertools.islice(it, batch_size))

def slugify(text: str) -> str:
    # Normalize to ASCII
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    # Lowercase, replace non-alphanum with underscores, collapse repeats
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    # Pinecone IDs must be ≤ 256 chars
    return text[:256] or "unknown"

def create_vector_database(
    json_path: str = "movies.json",
    index_name: str = "movies2",
    namespace: str = "example-namespace",
    batch_size: int = 95
):
    # 1) Init Pinecone client with your key
    pc = Pinecone(api_key="pcsk_3R7ydo_J5XoRYLTZpVreUhC6UuUbjPfB258sBdqFX8VKNi9LnCJCgeugyPbwezXB6my4wP")

    # 2) Load your JSON: [{"id": title, "text": descriptors}, …]
    with open(json_path, "r", encoding="utf-8") as f:
        raw_records = json.load(f)

    # 3) Slugify IDs and rebuild records
    records = []
    for rec in raw_records:
        safe_id = slugify(rec["id"])
        records.append({
            "id": safe_id,
            "text": rec["text"]
        })

    # 4) Create or connect to the index
    if not pc.has_index(index_name):
        print(f"Creating index '{index_name}' with integrated embeddings…")
        pc.create_index_for_model(
            name=index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "text"}
            }
        )
        time.sleep(10)  # wait for index to be ready
    else:
        print(f"Index '{index_name}' exists—connecting…")

    ix = pc.Index(index_name)

    # 5) Upsert in batches
    total = 0
    for batch in chunks(records, batch_size):
        ix.upsert_records(namespace=namespace, records=batch)
        total += len(batch)
        print(f"Upserted {total}/{len(records)} records")
        time.sleep(1)

    print(f"✅ Done: upserted {total} records into '{index_name}' namespace '{namespace}'")

if __name__ == "__main__":
    create_vector_database()
