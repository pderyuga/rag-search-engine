import os
from typing import TypedDict, Optional

from dotenv import load_dotenv
from google import genai

from .search_utils import DEFAULT_SEARCH_LIMIT

load_dotenv()
project = os.environ.get("GEMINI_PROJECT")
location = os.environ.get("GEMINI_LOCATION")

client = genai.Client(vertexai=True, project=project, location=location)
model_name = "gemini-2.0-flash-001"


class RRFSearchResult(TypedDict):
    id: int
    title: str
    description: str
    bm_25_rank: int
    semantic_rank: int
    rrf_score: float
    rank: int


class RankedRRFSearchResult(TypedDict):
    id: int
    title: str
    description: str
    bm_25_rank: int
    semantic_rank: int
    rrf_score: float
    rank: int


def individual_rerank(
    query: str,
    results: list[RRFSearchResult],
    limit: Optional[int] = DEFAULT_SEARCH_LIMIT,
) -> list[RankedRRFSearchResult]:
    for doc in results:
        prompt = f"""Rate how well this movie matches the search query.

Query: "{query}"
Movie: {doc.get("title", "")} - {doc.get("description", "")}

Consider:
- Direct relevance to query
- User intent (what they're looking for)
- Content appropriateness

Rate 0-10 (10 = perfect match).
Give me ONLY the number in your response, no other text or explanation.

Score:"""

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
        )
        stripped_response = (response.text or "").strip().strip('"')
        numeric_response = int(stripped_response)
        doc["rank"] = numeric_response

    sorted_results = sorted(
        results,
        key=lambda item: item["rank"],
        reverse=True,
    )

    top_results = sorted_results[:limit]

    return top_results


def rerank_results(
    query: str,
    results: list[RRFSearchResult],
    method: Optional[str] = None,
    limit: Optional[int] = DEFAULT_SEARCH_LIMIT,
) -> list[RankedRRFSearchResult]:
    match method:
        case "individual":
            return individual_rerank(query, results, limit)

        case _:
            return results
