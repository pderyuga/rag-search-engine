from .search_utils import (
    load_movies,
    load_golden_dataset,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_RRF_K,
)
from .semantic_search import SemanticSearch
from .hybrid_search import HybridSearch


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

        test_case["docs"] = titles
        test_case["precision"] = precision

    return test_cases
