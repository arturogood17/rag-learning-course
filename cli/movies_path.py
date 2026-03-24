import os
from pathlib import Path


DEFAULT_LIMIT = 5

parent = Path(os.path.dirname(os.path.realpath(__file__))).parent

file = os.path.join(parent, "data", "movies.json")

stop_words_file = os.path.join(parent, "data", "stopwords.txt")

cache_index = os.path.join(parent, "cache", "index.pkl")

cache_docmap = os.path.join(parent, "cache", "docmap.pkl")

cache_term_frequency = os.path.join(parent, "cache", "term_frequencies.pkl")

BM25_k1 = 1.5

