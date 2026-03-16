import json, string
from movies_path import *

def search(query: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    with open(file) as f:
        data = json.load(f)
    movies = []
    processed_query = stop_words(tokenization(text_processor(query)))
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
    return list(filter(None, s.split(" ")))

def stop_words(l: list[str]) -> list[str]:
    with open(stop_words_file) as f:
        content = f.read()
    list_sw= content.splitlines()
    new_tokens= []
    for t in l:
        if t not in list_sw:
            new_tokens.append(t)
    return new_tokens