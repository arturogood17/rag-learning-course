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

    def generate_embedding(self, text) ->list[float]:
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
    
    def search(self, query, limit):
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_vector = self.generate_embedding(query)
        similarity_list = []
        for i in range(len(self.embeddings)):
            similarity = cosine_similarity(query_vector, self.embeddings[i])
            similarity_list.append((similarity, self.documents[i]))
        sorted_list = sorted(similarity_list, key= lambda x: x[0], reverse=True)
        search_results = []
        for i in range(limit):
            search_results.append({"score": sorted_list[i][0], "title": sorted_list[i][1]['title'], "description": sorted_list[i][1]['description']})
        return search_results


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

def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def search(query: str, limit: int):
    embedder = SemanticSearch()
    with open(file) as f:
        movies = json.load(f)
    embedder.load_or_create_embeddings(movies['movies'])
    search_results = embedder.search(query, limit)
    for i, v in enumerate(search_results):
        print(f'{i+1}. {v["title"]} (score: {v["score"]})')
        print()
        print(v["description"])
        print()


def chunk(doc: str, chunk_size: int):
    split_doc = doc.split()
    buffer_doc = []
    chunks = []
    n = 0
    while True:
      if n >= len(split_doc):
        break
      buffer_doc = split_doc[n:chunk_size+n]
      chunks.append(buffer_doc)
      n += chunk_size
    chunk_printing(chunks, len(doc))


def chunk_printing(chunks: list[list], total_len: int):
    print(f"Chunking {total_len} characters")
    for i, v in enumerate(chunks, 1):
        print(f"{i}. {" ".join(v)}")
