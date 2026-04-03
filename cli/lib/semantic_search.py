from sentence_transformers import SentenceTransformer

class SemanticSearch:
    def __init__(self):
       self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Texto vacío")
        return self.model.encode([text])[0]



def verify_model():
    embedder = SemanticSearch()
    print("Model loaded:" , embedder.model)
    print("Max sequence length:" , embedder.model.max_seq_length)

def embed_text(text):
    embedder = SemanticSearch()
    embedding = embedder.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")
