import json, os
from pathlib import Path

def search(query: str) -> list[dict]:
    file = os.path.join(Path.cwd(), "data/movies.json")
    print(file)
    with open(file) as f:
        data = json.load(f)
    movies = []
    for x in data["movies"]:
        if query in x["title"]:
            movies.append(x)
    return movies