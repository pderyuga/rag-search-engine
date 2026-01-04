import os
import numpy as np
from typing import TypedDict
from sentence_transformers import SentenceTransformer
from .search_utils import (
    load_movies,
    MOVIE_EMBEDDINGS_PATH,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_CHUNK_SIZE,
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


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE):
    word_list = text.split()

    chunks: list[list[str]] = []
    # Step through the word_list in increments of 'chunk_size'
    for i in range(0, len(word_list), chunk_size):
        # Extract a chunk from index i to i+chunk_size
        chunk = word_list[i : i + chunk_size]
        chunks.append(chunk)

    text_chunks: list[str] = []
    for chunk in chunks:
        text_chunk = " ".join(chunk)
        text_chunks.append(text_chunk)

    print(f"Chunking {len(text)} characters")
    for index, text_chunk in enumerate(text_chunks, 1):
        print(f"{index}. {text_chunk}")
