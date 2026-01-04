import os
import numpy as np
from typing import TypedDict
from sentence_transformers import SentenceTransformer
from .search_utils import load_movies, MOVIE_EMBEDDINGS_PATH


class Movie(TypedDict):
    id: int
    title: str
    description: str


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


def verify_model():
    semantic_search = SemanticSearch()
    print(f"Model loaded: {semantic_search.model}")
    print(f"Max sequence length: {semantic_search.model.max_seq_length}")


def embed_text(text: str):
    semantic_search = SemanticSearch()
    embedding = semantic_search.generate_embedding(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")


def verify_embeddings():
    semantic_search = SemanticSearch()
    documents = load_movies()
    embeddings = semantic_search.load_or_create_embeddings(documents)

    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )
