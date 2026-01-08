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
from .hybrid_search import HybridSearch, RRFSearchResult
from .results_reranking import RRFSearchResult, RankedRRFSearchResult

load_dotenv()
project = os.environ.get("GEMINI_PROJECT")
location = os.environ.get("GEMINI_LOCATION")

client = genai.Client(vertexai=True, project=project, location=location)
model_name = "gemini-2.0-flash-001"


def evaluate_command(limit: int = DEFAULT_SEARCH_LIMIT):
    test_cases = load_golden_dataset()
    movies = load_movies()

    semantic_search = SemanticSearch()
    semantic_search.load_or_create_embeddings(movies)
    hybrid_search = HybridSearch(movies)

    for test_case in test_cases:
        results = hybrid_search.rrf_search(test_case["query"], DEFAULT_RRF_K, limit)
        titles = [result["title"] for result in results]

        relevant_titles = []
        for title in titles:
            if title in test_case["relevant_docs"]:
                relevant_titles.append(title)

        total_retrieved = len(titles)
        relevant_retrieved = len(relevant_titles)
        precision = relevant_retrieved / total_retrieved

        total_relevant = len(test_case["relevant_docs"])
        recall = relevant_retrieved / total_relevant

        f1 = 2 * (precision * recall) / (precision + recall)

        test_case["docs"] = titles
        test_case["precision"] = precision
        test_case["recall"] = recall
        test_case["f1_score"] = f1

    return test_cases


def llm_judge_results(
    query: str, results: list[RRFSearchResult] | list[RankedRRFSearchResult]
):
    formatted_results = str(
        [f"{result["title"]} - {result["description"]}" for result in results]
    )
    prompt = f"""Rate how relevant each result is to this query on a 0-3 scale:

Query: "{query}"

Results:
{chr(10).join(formatted_results)}

Scale:
- 3: Highly relevant
- 2: Relevant
- 1: Marginally relevant
- 0: Not relevant

Do NOT give any numbers out than 0, 1, 2, or 3.

Return ONLY the scores in the same order you were given the documents. Return a valid JSON list, nothing else. For example:

[2, 0, 3, 2, 0, 1]"""

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
    )

    stripped_response = (response.text or "").strip().strip('"')
    scores_list: list[int] = json.loads(stripped_response)

    for index, result in enumerate(results):
        result["llm_score"] = scores_list[index]
        print(f"{index + 1}. {result["title"]}: {result["llm_score"]}/3")
