from sentence_transformers import SentenceTransformer
import numpy as np
from utils import numpy_embeddings, file, chunk_embeddings, chunk_metadata
import os, json, re

class SemanticSearch:
    def __init__(self, model_name= "all-MiniLM-L6-v2"):
       self.model = SentenceTransformer(model_name)
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
    
    def search(self, query, limit) -> list[dict]:
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        query_vector = self.generate_embedding(query)
        similarity_list = []
        for i in range(len(self.embeddings)):
            similarity = cosine_similarity(query_vector, self.embeddings[i])
            similarity_list.append((similarity, self.documents[i]))
        similarity_list.sort(key= lambda x: x[0], reverse=True)
        search_results = []
        for score, doc in similarity_list[:limit]:
            search_results.append({"score": score, "title": doc['title'], "description": doc['description']})
        return search_results
    

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name = "all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, document):
        self.documents = document
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        all_chunks = []
        chunks_metadata = []
        semantic_chunks = []
        for index, doc in enumerate(self.documents):
            if doc['description'].strip():
                semantic_chunks = semantic_chunk(doc['description'].strip(), 4, 1)
                for i, c in enumerate(semantic_chunks):
                    all_chunks.append(c)
                    chunks_metadata.append({"movie_idx": index, "chunk_idx": i, "total_chunks":len(semantic_chunks),})
        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunks_metadata
        np.save(chunk_embeddings, self.chunk_embeddings)
        with open(chunk_metadata, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        return self.chunk_embeddings
    
    def search_chunks(self, query: str, limit: int = 10):
        embedded_query = self.generate_embedding(query)
        score = []
        for index, embedding in enumerate(self.chunk_embeddings):
            cs = cosine_similarity(embedded_query, embedding)
            score.append({"chunk_idx": self.chunk_metadata[index]["chunk_idx"], "movie_idx": self.chunk_metadata[index]["movie_idx"], "score": cs,})
        movie_score = {}
        for s in score:
            if s["movie_idx"] not in movie_score or s["score"] > movie_score[s["movie_idx"]]:
                movie_score[s["movie_idx"]] = s["score"]
        ordered_movies = sorted(movie_score.items(), key = lambda x: x[1] , reverse=True)
        results = []
        for v in ordered_movies[:limit]:
            movie_index = v[0]
            score_val = v[1]
            results.append({   
                "id": self.documents[movie_index]["id"],
                "title": self.documents[movie_index]["title"],
                "document": self.documents[movie_index]["description"][:100],
                "score": round(score_val, 3),
                "metadata": {},
            })
        return results
        
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc['id']] = doc
        if os.path.exists(chunk_embeddings) and os.path.exists(chunk_metadata):
            self.chunk_embeddings = np.load(chunk_embeddings)
            with open(chunk_metadata, "r") as f:
                dict_metadata = json.load(f)
                self.chunk_metadata = dict_metadata["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents)

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


def chunk(doc: str, chunk_size: int, overlap: int):
    split_doc = doc.split()
    chunks = []
    i = 0
    while i < len(split_doc):
      if len(split_doc[i:chunk_size+i]) <= overlap:
        break
      chunks.append(split_doc[i:chunk_size+i])
      i += chunk_size - overlap
    chunk_printing(chunks, len(doc))


def chunk_printing(chunks: list[str], total_len: int):
    # print(f"Chunking {total_len} characters")
    for i, v in enumerate(chunks, 1):
       print(f"{i}. {v}")

def semantic_chunk(text: str, max_size_chunk: int, overlap: int) -> list[str]:
    text = text.strip()
    chunks = []
    if not text:
        chunk_printing([" "], 0)
    text_split = re.split(r'(?<=[.!?])\s+', text)
    if len(text_split) == 1 and not text_split[0].endswith(".") and not text_split[0].endswith("!") and not text_split[0].endswith("?"):
        text = " ".join(text_split[0]).strip()
        if len(text) > 0:
            chunks.append(text)
    i = 0
    while i < len(text_split):
        if len(text_split[i:i+max_size_chunk]) <= overlap:
            break
        chunk = " ".join(text_split[i:i+max_size_chunk]).strip()
        if len(chunk) > 0:
            chunks.append(chunk)
        i += max_size_chunk - overlap
    return chunks