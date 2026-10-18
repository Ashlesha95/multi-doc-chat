import os
from functools import lru_cache
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Persistent folder inside your project (Windows won't clean it like Temp)
MODEL_CACHE_DIR = os.path.abspath("./model_cache")
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)


@lru_cache(maxsize=1)
def get_embeddings():
    """One shared embedding model, loaded once and used by ingest and rag."""
    return FastEmbedEmbeddings(
        model_name=EMBED_MODEL,
        max_length=256,
        threads=os.cpu_count() or 4,
        cache_dir=MODEL_CACHE_DIR,
    )