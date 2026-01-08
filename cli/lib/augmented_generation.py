import os
import json
from dotenv import load_dotenv
from google import genai

from .search_utils import (
    load_movies,
    load_golden_dataset,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_RRF_K,
)
from .semantic_search import SemanticSearch
from .hybrid_search import HybridSearch
from .results_reranking import RRFSearchResult, RankedRRFSearchResult

load_dotenv()
project = os.environ.get("GEMINI_PROJECT")
location = os.environ.get("GEMINI_LOCATION")

client = genai.Client(vertexai=True, project=project, location=location)
model_name = "gemini-2.0-flash-001"


def rag_command(query: str, k: int = DEFAULT_RRF_K, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    results = hybrid_search.rrf_search(query, DEFAULT_RRF_K, limit)

    docs = str([f"{result["title"]} - {result["description"]}" for result in results])

    prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.

Query: {query}

Documents:
{docs}

Provide a comprehensive answer that addresses the query:"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    response_text = (response.text or "").strip().strip('"')

    print("Search Results:")
    for result in results:
        print(f" - {result["title"]}")
    print()
    print("RAG Response:")
    print(response_text)
