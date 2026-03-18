#!/usr/bin/env python3

import argparse
from invertex_index import *



def pretty_printing_query_result(query: str, result: list[dict]):
    print("Searching for:", query)
    for i in range(len(result)):
        print(f"{i + 1}. {result[i]["title"]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    build_parser = subparsers.add_parser("build", help="Gives ID of docs where a word appears")
    search_parser.add_argument("query", type=str, help="Search query")



    args = parser.parse_args()

    Inverted_Index_Search = InvertedIndex()

    match args.command:
        case "search":
            try:
                Inverted_Index_Search.load()
            except:
                print("Error al cargar el archivo")
                return
            results = search_helper(Inverted_Index_Search, args.query)
            for r in results:
                print(f"{r['id']} - {r['title']}")
        case "build":
            Inverted_Index_Search.build()
            Inverted_Index_Search.save()
        case _:
            parser.print_help()

    

if __name__ == "__main__":
    main()