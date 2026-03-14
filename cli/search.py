import json
from movies_path import *

def search(query: str, limit: int = DEFAULT_LIMIT) -> list[dict]:
    with open(file) as f:
        data = json.load(f)
    movies = []
    for x in data["movies"]:
        if query.lower() in x["title"].lower():
            movies.append(x)
        if len(movies) >= DEFAULT_LIMIT:
            break
    return movies