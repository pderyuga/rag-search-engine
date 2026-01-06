import os
from typing import TypedDict

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch

from .search_utils import (
    load_movies,
    INDEX_PATH,
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    SEARCH_LIMIT_MULTIPLIER,
    DOCUMENT_PREVIEW_LENGTH,
    SCORE_PRECISION,
)


class Movie(TypedDict):
    id: int
    title: str
    description: str


class HybridSearchScore(TypedDict):
    doc: Movie
    norm_bm25_score: float
    norm_semantic_score: float
    hybrid_score: float


class HybridSearchResult(TypedDict):
    id: int
    title: str
    description: str
    bm25_score: float
    semantic_score: float
    hybrid_score: float


class HybridSearch:
    def __init__(self, documents):
        self.documents: list[Movie] = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)

        self.idx = InvertedIndex()
        if not os.path.exists(INDEX_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query, limit)

    def weighted_search(
        self,
        query: str,
        alpha: float = DEFAULT_ALPHA,
        limit: int = DEFAULT_SEARCH_LIMIT,
    ):
        bm25_results = self._bm25_search(query, limit * SEARCH_LIMIT_MULTIPLIER)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * SEARCH_LIMIT_MULTIPLIER
        )

        bm25_scores = list(map(lambda result: result["score"], bm25_results))
        semantic_scores = list(map(lambda result: result["score"], semantic_results))

        bm25_doc_ids = list(map(lambda result: result["id"], bm25_results))
        semantic_doc_ids = list(map(lambda result: result["id"], semantic_results))

        # Normalize all the keyword and semantic scores using the normalize method
        normalized_bm25_scores = normalize_scores(bm25_scores)
        normalized_semantic_scores = normalize_scores(semantic_scores)

        normalized_bm25_scores_map = dict(zip(bm25_doc_ids, normalized_bm25_scores))
        normalized_semantic_scores_map = dict(
            zip(semantic_doc_ids, normalized_semantic_scores)
        )

        # Create a dictionary mapping document IDs to the documents themselves and their keyword and semantic scores
        normalized_scores_map: dict[int, HybridSearchScore] = {}
        for doc in self.documents:
            doc_id = doc["id"]
            normalized_scores_map[doc_id] = {}

            bm25_score = 0.0
            semantic_score = 0.0

            if doc_id in normalized_bm25_scores_map:
                bm25_score = normalized_bm25_scores_map[doc_id]
            if doc_id in normalized_semantic_scores_map:
                semantic_score = normalized_semantic_scores_map[doc_id]

            normalized_scores_map[doc_id]["doc"] = doc
            normalized_scores_map[doc_id]["norm_bm25_score"] = bm25_score
            normalized_scores_map[doc_id]["norm_semantic_score"] = semantic_score
            # Add a third score (the hybrid score) to each document
            normalized_scores_map[doc_id]["hybrid_score"] = hybrid_score(
                bm25_score,
                semantic_score,
                alpha,
            )

        # sort the documents by score in descending order
        sorted_scores = sorted(
            normalized_scores_map.items(),
            key=lambda item: item[1]["hybrid_score"],
            reverse=True,
        )

        # return the top limit documents along with their scores
        top_scores = sorted_scores[:limit]

        # Format the results and limit the movie description to the first 100 characters.
        results: list[HybridSearchResult] = []
        for doc_id, result in top_scores:
            doc = result["doc"]
            search_result: HybridSearchResult = {}
            search_result["id"] = doc_id
            search_result["title"] = doc["title"]
            search_result["description"] = doc["description"][:DOCUMENT_PREVIEW_LENGTH]
            search_result["bm25_score"] = round(
                result["norm_bm25_score"], SCORE_PRECISION
            )
            search_result["semantic_score"] = round(
                result["norm_semantic_score"], SCORE_PRECISION
            )
            search_result["hybrid_score"] = round(
                result["hybrid_score"], SCORE_PRECISION
            )
            results.append(search_result)
        # Return the final list of results.
        return results

    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")


def normalize_scores(scores: list[float]):
    if scores is None or len(scores) == 0:
        return []

    min_score = min(scores)
    max_score = max(scores)

    normalized_scores: list[float] = []

    if min_score == max_score:
        for score in scores:
            normalized_scores.append(1.0)
    else:
        for score in scores:
            normalized_score = (score - min_score) / (max_score - min_score)
            normalized_scores.append(normalized_score)

    return normalized_scores


def hybrid_score(
    bm25_score: float, semantic_score: float, alpha: float = DEFAULT_ALPHA
):
    return alpha * bm25_score + (1 - alpha) * semantic_score


def weighted_search_command(
    query: str, alpha: float = DEFAULT_ALPHA, limit: float = DEFAULT_SEARCH_LIMIT
):
    documents = load_movies()
    search_instance = HybridSearch(documents)

    results = search_instance.weighted_search(query, alpha, limit)

    print(f"Query: {query}")
    print(f"Alpha: {alpha}")
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result["title"]}")
        print(f"Hybrid Score: {result["hybrid_score"]:.4f}")
        print(
            f"BM25: {result["bm25_score"]:.4f}, Semantic: {result["semantic_score"]:.4f}"
        )
        print(f"   {result["description"]}...")
