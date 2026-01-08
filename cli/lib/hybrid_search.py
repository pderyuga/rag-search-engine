import os
from typing import TypedDict, Optional

from .keyword_search import InvertedIndex
from .semantic_search import ChunkedSemanticSearch
from .query_enhancement import enhance_query
from .results_reranking import rerank_results

from .search_utils import (
    load_movies,
    INDEX_PATH,
    DEFAULT_ALPHA,
    DEFAULT_SEARCH_LIMIT,
    SEARCH_LIMIT_MULTIPLIER,
    DOCUMENT_PREVIEW_LENGTH,
    SCORE_PRECISION,
    DEFAULT_RRF_K,
    RERANKING_SEARCH_LIMIT_MULTIPLIER,
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


class RRFSearchScore(TypedDict):
    doc: Movie
    bm_25_rank: int
    semantic_rank: int
    rrf_score: float


class RRFSearchResult(TypedDict):
    id: int
    title: str
    description: str
    bm_25_rank: int
    semantic_rank: int
    rrf_score: float


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

    def rrf_search(
        self, query: str, k: int = DEFAULT_RRF_K, limit: int = DEFAULT_SEARCH_LIMIT
    ):
        bm25_results = self._bm25_search(query, limit * SEARCH_LIMIT_MULTIPLIER)
        semantic_results = self.semantic_search.search_chunks(
            query, limit * SEARCH_LIMIT_MULTIPLIER
        )

        # sort results to get documents ranks
        sorted_bm25_results = sorted(
            bm25_results,
            key=lambda item: item["score"],
            reverse=True,
        )
        sorted_semantic_results = sorted(
            semantic_results,
            key=lambda item: item["score"],
            reverse=True,
        )

        # get document ranks as a list
        bm25_doc_ids_ranked = list(
            map(lambda result: result["id"], sorted_bm25_results)
        )
        semantic_doc_ids_ranked = list(
            map(lambda result: result["id"], sorted_semantic_results)
        )

        # Create a dictionary mapping document IDs to the documents themselves and their BM25 and semantic ranks
        rrf_scores_map: dict[int, RRFSearchScore] = {}
        for doc in self.documents:
            doc_id = doc["id"]
            rrf_scores_map[doc_id] = {}

            bm25_rank = None
            semantic_rank = None

            if doc_id in bm25_doc_ids_ranked:
                bm25_rank = bm25_doc_ids_ranked.index(doc_id)
            if doc_id in semantic_doc_ids_ranked:
                semantic_rank = semantic_doc_ids_ranked.index(doc_id)

            rrf_scores_map[doc_id]["doc"] = doc
            rrf_scores_map[doc_id]["bm25_rank"] = bm25_rank
            rrf_scores_map[doc_id]["semantic_rank"] = semantic_rank
            # For each document, calculate the RRF score using the rrf_score function and add that score to each document as well
            bm25_rrf_score = rrf_score(bm25_rank, k)
            semantic_rrf_score = rrf_score(semantic_rank, k)
            rrf_scores_map[doc_id]["rrf_score"] = bm25_rrf_score + semantic_rrf_score

        # sort the documents by score in descending order
        sorted_scores = sorted(
            rrf_scores_map.items(),
            key=lambda item: item[1]["rrf_score"],
            reverse=True,
        )

        # return the top limit documents along with their scores
        top_scores = sorted_scores[:limit]

        # Format the results and limit the movie description to the first 100 characters.
        results: list[RRFSearchResult] = []
        for doc_id, result in top_scores:
            doc = result["doc"]
            search_result: RRFSearchResult = {}
            search_result["id"] = doc_id
            search_result["title"] = doc["title"]
            search_result["description"] = doc["description"][:DOCUMENT_PREVIEW_LENGTH]
            search_result["bm25_rank"] = result["bm25_rank"]
            search_result["semantic_rank"] = result["semantic_rank"]
            search_result["rrf_score"] = round(result["rrf_score"], SCORE_PRECISION)
            results.append(search_result)
        # Return the final list of results.
        return results


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


def rrf_score(rank: int, k: int = DEFAULT_RRF_K):
    if rank is None:
        return 0
    return 1 / (k + rank)


def rrf_search_command(
    query: str,
    k: int = DEFAULT_RRF_K,
    limit: float = DEFAULT_SEARCH_LIMIT,
    enhance: Optional[str] = None,
    rerank_method: Optional[str] = None,
):
    documents = load_movies()
    search_instance = HybridSearch(documents)

    original_query = query
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query, method=enhance)
        query = enhanced_query

    search_limit = limit * RERANKING_SEARCH_LIMIT_MULTIPLIER if rerank_method else limit
    results = search_instance.rrf_search(query, k, search_limit)

    reranked = False
    if rerank_method:
        results = rerank_results(query, results, method=rerank_method, limit=limit)
        reranked = True

    if enhance:
        print(f"Enhanced query ({enhance}): '{original_query}' -> '{enhanced_query}'")
    else:
        print(f"Query: {query}")
    print(f"k: {k}")
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result["title"]}")
        if reranked:
            print(f"Rerank Score: {result["rank"]}/10")
        print(f"RRF Score: {result["rrf_score"]:.4f}")
        print(
            f"BM25 Rank: {result["bm25_rank"]}, Semantic Rank: {result["semantic_rank"]}"
        )
        print(f"   {result["description"]}...")

    return results
