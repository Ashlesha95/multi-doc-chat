import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings

load_dotenv()  # so RERANK_MODEL etc. from .env are visible when this module loads

EMBED_MODEL = "BAAI/bge-small-en-v1.5"

# Cross-encoder used to rerank retrieved passages (runs locally via FastEmbed).
# Small + fast by default (~80 MB). For higher accuracy try "BAAI/bge-reranker-base" (~1 GB)
# or "Xenova/ms-marco-MiniLM-L-12-v2" (~120 MB).
RERANK_MODEL = os.getenv("RERANK_MODEL", "Xenova/ms-marco-MiniLM-L-6-v2")

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


@lru_cache(maxsize=1)
def get_reranker():
    """One shared cross-encoder reranker, loaded lazily on first use.
    Needs a recent fastembed (pip install -U fastembed)."""
    from fastembed.rerank.cross_encoder import TextCrossEncoder

    return TextCrossEncoder(
        model_name=RERANK_MODEL,
        cache_dir=MODEL_CACHE_DIR,
        threads=os.cpu_count() or 4,
    )