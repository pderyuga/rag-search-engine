import os
import json
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
        key=lambda result: result["rank"],
        reverse=True,
    )

    top_results = sorted_results[:limit]

    return top_results


def batch_rerank(
    query: str,
    results: list[RRFSearchResult],
    limit: Optional[int] = DEFAULT_SEARCH_LIMIT,
) -> list[RankedRRFSearchResult]:
    doc_list_str = str(results)

    prompt = f"""Rank these movies by relevance to the search query.

Query: "{query}"

Movies:
{doc_list_str}

Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

[75, 12, 34, 2, 1]
"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    stripped_response = (response.text or "").strip().strip('"')
    scores_list: list[int] = json.loads(stripped_response)

    for doc in results:
        doc["rank"] = scores_list.index(doc["id"] + 1)

    sorted_results = sorted(results, key=lambda result: result["rank"], reverse=False)

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

        case "batch":
            return batch_rerank(query, results, limit)

        case _:
            return results
