import json, string
from movies_path import *

def search(query: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    with open(file) as f:
        data = json.load(f)
    movies = []
    processed_query = text_processor(query)
    for x in data["movies"]:
        processed_title = text_processor(x["title"])
        if processed_query in processed_title:
            movies.append(x)
        if len(movies) >= limit:
            break
    return movies


def text_processor(s: str) -> str:
    return s.lower().translate(str.maketrans('', '', string.punctuation))