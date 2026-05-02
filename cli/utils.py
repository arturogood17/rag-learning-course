import os, json
from pathlib import Path

K_VALUE = 60

DEFAULT_LIMIT = 5

parent = Path(os.path.dirname(os.path.realpath(__file__))).parent

file = os.path.join(parent, "data", "movies.json")

stop_words_file = os.path.join(parent, "data", "stopwords.txt")

cache_index = os.path.join(parent, "cache", "index.pkl")

cache_docmap = os.path.join(parent, "cache", "docmap.pkl")

cache_term_frequency = os.path.join(parent, "cache", "term_frequencies.pkl")

cache_doc_lenghts = os.path.join(parent, "data", "doc_lengths.pkl")

BM25_k1 = 1.5

BM25_B = 0.75

numpy_embeddings = os.path.join(parent, "cache", "movie_embeddings.npy")

chunk_embeddings = os.path.join(parent, "cache", "chunk_embeddings.npy")

chunk_metadata = os.path.join(parent, "cache", "chunk_metadata.json")

golden_dataset_path = os.path.join(parent, "data", "golden_dataset.json")

def load_file_json(path: str) -> any:
    with open(path) as f:
        file = json.load(f)
    return file