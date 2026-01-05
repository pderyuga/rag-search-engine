import os
import re
import json
import numpy as np
from typing import TypedDict
from sentence_transformers import SentenceTransformer
from .search_utils import (
    load_movies,
    MOVIE_EMBEDDINGS_PATH,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    CHUNK_EMBEDDINGS_PATH,
    CHUNK_METADATA_PATH,
    SCORE_PRECISION,
    DOCUMENT_PREVIEW_LENGTH,
)


class Movie(TypedDict):
    id: int
    title: str
    description: str


class SearchResult(TypedDict):
    id: int
    title: str
    description: str
    score: float


class SemanticSearch:

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents: list[Movie] = None
        self.document_map: dict[int, Movie] = {}

    def generate_embedding(self, text: str):
        if not text or not text.strip():
            raise ValueError("Cannot generate embedding for empty text")

        result = self.model.encode([text])
        embedding = result[0]
        return embedding

    def build_embeddings(self, documents: list[Movie]):
        self.documents = documents

        doc_list: list[str] = []
        for document in self.documents:
            self.document_map[document["id"]] = document
            doc_list.append(f"{document['title']}: {document['description']}")

        self.embeddings = self.model.encode(doc_list, show_progress_bar=True)

        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)

        return self.embeddings

    def load_or_create_embeddings(self, documents: list[Movie]):
        self.documents = documents

        doc_list: list[str] = []
        for document in self.documents:
            self.document_map[document["id"]] = document
            doc_list.append(f"{document['title']}: {document['description']}")

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)

            if (
                self.embeddings is not None
                and self.documents is not None
                and len(self.embeddings) == len(self.documents)
            ):
                return self.embeddings

        return self.build_embeddings(documents)

    def search(self, query: str, limit: int = DEFAULT_SEARCH_LIMIT):
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )

        query_embedding = self.generate_embedding(query)

        documents: list[tuple[float, Movie]] = []
        for index, embedding in enumerate(self.embeddings):
            similarity_score = cosine_similarity(query_embedding, embedding)
            documents.append((similarity_score, self.document_map[index + 1]))

        sorted_documents = sorted(documents, key=lambda item: item[0], reverse=True)

        top_documents = sorted_documents[:limit]

        results: list[SearchResult] = []
        for score, doc in top_documents:
            search_result: SearchResult = {}
            search_result["id"] = doc["id"]
            search_result["title"] = doc["title"]
            search_result["description"] = doc["description"]
            search_result["score"] = score
            results.append(search_result)

        return results


class ChunkMetadata(TypedDict):
    movie_idx: int
    chunk_idx: int
    total_chunks: int


class ChunkScore(TypedDict):
    movie_idx: int
    chunk_idx: int
    score: float


class SemanticSearchResult(TypedDict):
    id: int
    title: str
    description: str
    score: float
    metadata: ChunkMetadata


class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[Movie]):
        self.documents = documents

        for document in self.documents:
            self.document_map[document["id"]] = document

        all_chunks: list[str] = []
        chunk_metadata: list[ChunkMetadata] = []
        for doc_idx, document in enumerate(self.documents):
            if document["description"] is None or document["description"] == "":
                pass

            chunks = semantic_chunk_text(
                document["description"],
                DEFAULT_SEMANTIC_CHUNK_SIZE,
                DEFAULT_CHUNK_OVERLAP,
            )

            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                metadata: ChunkMetadata = {}
                metadata["movie_idx"] = doc_idx
                metadata["chunk_idx"] = chunk_idx
                metadata["total_chunks"] = len(chunks)
                chunk_metadata.append(metadata)

        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)

        os.makedirs(os.path.dirname(CHUNK_METADATA_PATH), exist_ok=True)

        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings

    def load_or_create_chunk_embeddings(self, documents: list[Movie]):
        self.documents = documents

        for document in self.documents:
            self.document_map[document["id"]] = document

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)

            with open(CHUNK_METADATA_PATH, "r") as f:
                self.chunk_metadata = json.load(f)["chunks"]

            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)

    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call `load_or_create_chunk_embeddings` first."
            )

        # Generate an embedding of the query
        query_embedding = self.generate_embedding(query)

        # Populate an empty list to store "chunk score" dictionaries
        chunk_scores: list[ChunkScore] = []
        # For each chunk embedding
        for chunk_embedding_index, chunk_embedding in enumerate(self.chunk_embeddings):
            # Calculate the cosine similarity between the chunk embedding and the query embedding
            similarity_score = cosine_similarity(query_embedding, chunk_embedding)
            # Append a dictionary to the chunk score list with fields:
            chunk_score: ChunkScore = {}
            # chunk_idx: The index of the chunk within the document
            chunk_score["chunk_idx"] = self.chunk_metadata[chunk_embedding_index][
                "chunk_idx"
            ]
            # movie_idx: The index of the document in self.documents (you'll need to use self.chunk_metadata to map back to this)
            chunk_score["movie_idx"] = self.chunk_metadata[chunk_embedding_index][
                "movie_idx"
            ]
            # The cosine similarity score
            chunk_score["score"] = similarity_score
            chunk_scores.append(chunk_score)

        # Create an empty dictionary that maps movie indexes to their scores
        movie_scores: dict[int, float] = {}
        # For each chunk score,
        for chunk_score in chunk_scores:
            # if the movie_idx is not in the movie score dictionary yet,
            # or the new score is higher than the existing one,
            if (
                chunk_score["movie_idx"] not in movie_scores
                or chunk_score["score"] > movie_scores[chunk_score["movie_idx"]]
            ):
                # update the movie score dictionary with the new chunk score.
                movie_scores[chunk_score["movie_idx"]] = chunk_score["score"]

        # Sort the movie scores by score in descending order.
        sorted_movie_scores = sorted(
            movie_scores.items(), key=lambda item: item[1], reverse=True
        )

        # Filter down to the top `limit` movies.
        top_movie_scores = sorted_movie_scores[:limit]

        # Format the results and limit the movie description to the first 100 characters.
        results: list[SemanticSearchResult] = []
        for doc_id, score in top_movie_scores:
            doc = self.documents[doc_id]
            search_result: SemanticSearchResult = {}
            search_result["id"] = doc["id"]
            search_result["title"] = doc["title"]
            search_result["description"] = doc["description"][:DOCUMENT_PREVIEW_LENGTH]
            search_result["score"] = round(score, SCORE_PRECISION)
            search_result["metadata"] = self.chunk_metadata
            results.append(search_result)
        # Return the final list of results.
        return results


def verify_model():
    search_instance = SemanticSearch()
    print(f"Model loaded: {search_instance.model}")
    print(f"Max sequence length: {search_instance.model.max_seq_length}")


def embed_text(text: str):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )


def embed_query_text(query: str):
    search_instance = SemanticSearch()
    embedding = search_instance.generate_embedding(query)

    print(f"Query: {query}")
    print(f"First 5 dimensions: {embedding[:5]}")
    print(f"Shape: {embedding.shape}")


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


def semantic_search(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for index, search_result in enumerate(results):
        print(
            f"{index + 1}. {search_result["title"]} (Score: {search_result["score"]:.4f})\n{search_result["description"]}\n"
        )


def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    if overlap >= chunk_size:
        raise ValueError("Chunk size must be greater than overlap")

    word_list = text.split()

    chunks: list[list[str]] = []

    # Sliding window chunking with overlap
    i = 0
    # Continue while:
    #   1. We haven't reached the end (i < len(word_list))
    #   2. Remaining words exceed overlap (prevents redundant chunks)
    while i < len(word_list) and (len(word_list) - i) > overlap:
        # Extract chunk_size words starting at position i
        chunk = word_list[i : i + chunk_size]
        chunks.append(chunk)
        # Advance by (chunk_size - overlap) so next chunk overlaps
        # Example: chunk_size=5, overlap=2 → advance by 3
        i += chunk_size - overlap

    text_chunks: list[str] = []
    for chunk in chunks:
        text_chunk = " ".join(chunk)
        text_chunks.append(text_chunk)

    print(f"Chunking {len(text)} characters")
    for index, text_chunk in enumerate(text_chunks, 1):
        print(f"{index}. {text_chunk}")


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
):
    if overlap >= max_chunk_size:
        raise ValueError("Max chunk size must be greater than overlap")

    valid_text = text.strip()
    if not valid_text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", valid_text)

    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        sentences = [valid_text]

    chunks: list[list[str]] = []
    i = 0
    while i < len(sentences) and (i == 0 or (len(sentences) - i) > overlap):
        chunk = sentences[i : i + max_chunk_size]
        valid_chunk: list[str] = []
        for sentence in chunk:
            valid_chunk.append(sentence.strip())
        if not valid_chunk:
            continue
        chunks.append(valid_chunk)
        i += max_chunk_size - overlap

    sentence_chunks: list[str] = []
    for chunk in chunks:
        sentence_chunk = " ".join(chunk)
        sentence_chunks.append(sentence_chunk)

    print(f"Semantically chunking {len(text)} characters")
    for index, sentence_chunk in enumerate(sentence_chunks, 1):
        print(f"{index}. {sentence_chunk}")

    return sentence_chunks


def embed_chunks():
    search_instance = ChunkedSemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_chunk_embeddings(documents)

    print(f"Generated {len(embeddings)} chunked embeddings")


def search_chunked(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    search_instance = ChunkedSemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_chunk_embeddings(documents)

    results = search_instance.search_chunks(query, limit)

    print(f"Query: {query}")
    print("Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result["title"]} (score: {result["score"]:.4f})")
        print(f"   {result["description"]}...")
