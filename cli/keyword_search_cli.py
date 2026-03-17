#!/usr/bin/env python3

import argparse
from search import search_func
from invertex_index import InvertedIndex



def pretty_printing_query_result(query: str, result: list[dict]):
    print("Searching for:", query)
    for i in range(len(result)):
        print(f"{i + 1}. {result[i]["title"]}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")
    search_parser = subparsers.add_parser("build", help="Gives ID of where a word appears")

    args = parser.parse_args()

    match args.command:
        case "search":
            result = search_func(args.query)                      #Se hace así porque argparse convierte el add_argument("query") en un objeto         
            pretty_printing_query_result(args.query, result) #al que se puede acceder directamente
        case "build":
            Inverted_Index_Search = InvertedIndex()
            Inverted_Index_Search.build()
            Inverted_Index_Search.save()
            ids = list(Inverted_Index_Search.get_documents('merida'))
            print(f"First document for token 'merida' = {ids[0]}")
        case _:
            parser.print_help()

    

if __name__ == "__main__":
    main()