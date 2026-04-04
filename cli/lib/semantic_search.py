from sentence_transformers import SentenceTransformer
import numpy as np
from movies_path import numpy_embeddings, file
import os, json

class SemanticSearch:
    def __init__(self):
       self.model = SentenceTransformer("all-MiniLM-L6-v2")
       self.embeddings = None
       self.documents = None
       self.document_map = {}

    def generate_embedding(self, text):
        if not text or not text.strip():
            raise ValueError("Texto vacío")
        return self.model.encode([text])[0]
    
    def build_embeddings(self, documents: list[dict]) -> list[str]:
        self.documents = documents
        movies = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            title_doc = f"{doc['title']}: {doc['description']}"
            movies.append(title_doc)
        self.embeddings = self.model.encode(movies, show_progress_bar = True)
        save_np(self.embeddings)
        return self.embeddings
    
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        movies = []
        for doc in self.documents:
            self.document_map[doc['id']] = doc
            title_doc = f"{doc['title']}: {doc['description']}"
            movies.append(title_doc)
        if os.path.exists(numpy_embeddings):
            self.embeddings = np.load(numpy_embeddings)
            if len(self.embeddings) == len(self.documents):
                return self.embeddings
        self.build_embeddings(documents)


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

def save_np(l: list[any]):
    np.save(numpy_embeddings, l)

def verify_embeddings():
    embedder = SemanticSearch()
    with open(file) as f:
        movies = json.load(f)
    embeddings = embedder.load_or_create_embeddings(movies["movies"])
    print(f"Number of docs:   {len(movies["movies"])}")
    print(f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")

def embed_query_text(query):
    embedder = SemanticSearch()
    query_emb = embedder.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 3 dimensions: {query_emb[:3]}")
    print(f"Shape: {query_emb.shape}")

