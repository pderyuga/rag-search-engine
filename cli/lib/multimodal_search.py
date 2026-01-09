import os
from PIL import Image
from sentence_transformers import SentenceTransformer


class MultimodalSearch:
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image: str):
        if not os.path.exists(image):
            raise FileNotFoundError(f"Image file not found: {image}")

        img = Image.open(image)
        result = self.model.encode([img])
        embedding = result[0]
        return embedding


def verify_image_embedding(image: str):
    search_instance = MultimodalSearch()
    embedding = search_instance.embed_image(image)

    print(f"Embedding shape: {embedding.shape[0]} dimensions")
