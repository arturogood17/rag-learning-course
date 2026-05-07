from PIL import Image
from sentence_transformers import SentenceTransformer
from .semantic_search import cosine_similarity
from utils import file, load_file_json, DEFAULT_LIMIT

class MultimodalSearch:
    def __init__(self, docs: list[any], model_name = "clip-ViT-B-32") -> None:
        self.model = SentenceTransformer(model_name)
        self.docs = docs
        self.texts = [] #f"{doc['title']}: {doc['description']}"
        for doc in self.docs:
            self.texts.append(f"{doc['title']}: {doc['description']}")
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, path: str):
        image = Image.open(path)
        return self.model.encode([image])[0]
    
    def search_with_image(self, path_img: str):
        embedded_img = self.embed_image(path_img)
        results = []
        for index, txt_embedding in enumerate(self.text_embeddings):
            cosine_simil = cosine_similarity(embedded_img, txt_embedding)
            results.append({"doc_id": self.docs[index]["id"],
                            "title": self.docs[index]["title"],
                            "description": self.docs[index]["description"],
                            "cosine_similarity":  cosine_simil})
        return sorted(results, key= lambda x: x["cosine_similarity"], reverse=True)[:DEFAULT_LIMIT]


def verify_image_embedding(img_path: str):
    movies = load_file_json(file)
    if not img_path:
        return
    multi_m_search = MultimodalSearch(docs=movies["movies"])
    embedding = multi_m_search.embed_image(img_path)
    print(f"Embedding shape: {embedding.shape[0]} dimensions")

def image_search_command(img_path: str):
    movies = load_file_json(file)
    multi_m_search = MultimodalSearch(docs=movies["movies"])
    results = multi_m_search.search_with_image(img_path)
    return results