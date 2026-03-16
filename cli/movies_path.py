import os
from pathlib import Path


DEFAULT_LIMIT = 5

parent = Path(os.path.dirname(os.path.realpath(__file__))).parent

file = os.path.join(parent, "data", "movies.json")

stop_words_file = os.path.join(parent, "data", "stopwords.txt")
