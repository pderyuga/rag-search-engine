import json
import os

DEFAULT_SEARCH_LIMIT = 5

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "movies.json")
STOPWORDS_PATH = os.path.join(PROJECT_ROOT, "data", "stopwords.txt")
CACHE_PATH = os.path.join(PROJECT_ROOT, "cache")
INDEX_PATH = os.path.join(CACHE_PATH, "index.pkl")
DOCMAP_PATH = os.path.join(CACHE_PATH, "docmap.pkl")
TF_PATH = os.path.join(CACHE_PATH, "term_frequencies.pkl")


def load_movies() -> list[dict]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        content = f.read()
    stopwords = content.splitlines()
    return stopwords
