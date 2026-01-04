import json
import os

DEFAULT_SEARCH_LIMIT = 5
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_CHUNK_SIZE = 200
DEFAULT_CHUNK_OVERLAP = 0

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_PATH, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_PATH, "docmap.pkl")
TF_PATH = os.path.join(CACHE_PATH, "term_frequencies.pkl")
DOC_LENGTHS_PATH = os.path.join(CACHE_PATH, "doc_lengths.pkl")
MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_PATH, "movie_embeddings.npy")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        content = f.read()
    stopwords = content.splitlines()
    return stopwords
