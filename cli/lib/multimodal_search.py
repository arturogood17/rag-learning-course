from PIL import Image
from sentence_transformers import SentenceTransformer

class MultimodalSearch:
    def __init__(self, model_name = "clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)

    def embed_image(self, path: str):
        image = Image.open(path)
        return self.model.encode([image])[0]


def verify_image_embedding(img_path: str):
    if not img_path:
        return
    multi_m_search = MultimodalSearch()
    embedding = multi_m_search.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")