import os
from dotenv import load_dotenv
from google import genai

from .search_utils import (
    load_movies,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_RRF_K,
)
from .semantic_search import SemanticSearch
from .hybrid_search import HybridSearch

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

    results = hybrid_search.rrf_search(query, k, limit)

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


def summarize_command(
    query: str, k: int = DEFAULT_RRF_K, limit: int = DEFAULT_SEARCH_LIMIT
):
    movies = load_movies()

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    results = hybrid_search.rrf_search(query, k, limit)

    docs = str([f"{result["title"]} - {result["description"]}" for result in results])

    prompt = f"""
Provide information useful to this query by synthesizing information from multiple search results in detail.
The goal is to provide comprehensive information so that users know what their options are.
Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
This should be tailored to Hoopla users. Hoopla is a movie streaming service.
Query: {query}
Search Results:
{docs}
Provide a comprehensive 3-4 sentence answer that combines information from multiple sources:
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    response_text = (response.text or "").strip().strip('"')

    print("Search Results:")
    for result in results:
        print(f" - {result["title"]}")
    print()
    print("LLM Summary:")
    print(response_text)


def citations_command(
    query: str, k: int = DEFAULT_RRF_K, limit: int = DEFAULT_SEARCH_LIMIT
):
    movies = load_movies()

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    results = hybrid_search.rrf_search(query, k, limit)

    docs = str([f"{result["title"]} - {result["description"]}" for result in results])

    prompt = f"""Answer the question or provide information based on the provided documents.

This should be tailored to Hoopla users. Hoopla is a movie streaming service.

If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

Query: {query}

Documents:
{docs}

Instructions:
- Provide a comprehensive answer that addresses the query
- Cite sources using [1], [2], etc. format when referencing information
- If sources disagree, mention the different viewpoints
- If the answer isn't in the documents, say "I don't have enough information"
- Be direct and informative

Answer:"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    response_text = (response.text or "").strip().strip('"')

    print("Search Results:")
    for result in results:
        print(f" - {result["title"]}")
    print()
    print("LLM Answer:")
    print(response_text)
