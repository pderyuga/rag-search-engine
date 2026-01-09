import os
from PIL import Image
from sentence_transformers import SentenceTransformer

from .semantic_search import cosine_similarity, Movie, SearchResult
from .search_utils import load_movies, DEFAULT_SEARCH_LIMIT


class MultimodalSearch:

    def __init__(self, documents, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents: list[Movie] = documents
        self.texts = load_texts(self.documents)
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image: str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")

        img = Image.open(image)
        result = self.model.encode([img])
        embedding = result[0]
        return embedding

    def search_with_image(self, image: str, limit: int = DEFAULT_SEARCH_LIMIT):
        image_embedding = self.embed_image(image)

        docs_with_scores: list[SearchResult] = []
        for index, text_embedding in enumerate(self.text_embeddings):
            similarity_score = cosine_similarity(image_embedding, text_embedding)
            doc_with_score = self.documents[index]
            doc_with_score["score"] = similarity_score
            docs_with_scores.append(doc_with_score)

        sorted_documents = sorted(
            docs_with_scores, key=lambda item: item["score"], reverse=True
        )

        top_documents = sorted_documents[:limit]
        return top_documents


def load_texts(documents):
    texts: list[str] = []
    for doc in documents:
        texts.append(f"{doc['title']}: {doc['description']}")

    return texts


def verify_image_embedding(image: str):
    documents = load_movies()
    search_instance = MultimodalSearch(documents)
    embedding = search_instance.embed_image(image)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")


def image_search_command(image: str):
    documents = load_movies()
    search_instance = MultimodalSearch(documents)
    results = search_instance.search_with_image(image)

    return results
