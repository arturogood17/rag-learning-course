#!/usr/bin/env python3

import argparse, math
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
    tf_parser = subparsers.add_parser("tf", help="Gives term frequency")
    tf_parser.add_argument("id", type=int, help="Term frequency doc id")
    tf_parser.add_argument("term", type=str, help="Term frequency term to look up")
    search_parser.add_argument("query", type=str, help="Search query")
    idf_suparser = subparsers.add_parser("idf", help="Helps calculate the idf of a term")
    idf_suparser.add_argument("term", type=str, help="The term that is going to be look up in the dataset")



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

        case "idf":
            Inverted_Index_Search.load()
            idf = idf_func(args.term, Inverted_Index_Search)        
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")        
        case "tf":
            Inverted_Index_Search.load()
            print(Inverted_Index_Search.get_tf(args.id, args.term))
        case _:
            parser.print_help()

    

if __name__ == "__main__":
    main()