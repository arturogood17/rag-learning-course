import json, string
from movies_path import *
from nltk.stem import PorterStemmer

def search(query: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    with open(file) as f:
        data = json.load(f)
    movies = []
    processed_query = tokenization(text_processor(query))
    for m in data["movies"]:
        processed_title = text_processor(m["title"])
        for token in processed_query:
            if token in processed_title and not any(m["title"] == x["title"] for x in movies):
                movies.append(m)
        if len(movies) >= limit:
            break
    return movies


def text_processor(s: str) -> str:
    return s.lower().translate(str.maketrans('', '', string.punctuation))

def tokenization(s: str) -> list[str]:
    stemmer = PorterStemmer()
    tokens = list(filter(None, s.split(" ")))
    with open(stop_words_file) as f:
        content = f.read()
    list_sw= content.splitlines()
    new_tokens= []
    for t in tokens:
        if t not in list_sw:
            new_tokens.append(stemmer.stem(t))
    return new_tokens