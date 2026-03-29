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
    tfidf_suparser = subparsers.add_parser("tfidf", help="Helps calculate the tfidf of a term")
    tfidf_suparser.add_argument("doc_id", type=int, help="Document id")
    tfidf_suparser.add_argument("term", type=str, help="The term that is going to be look up in the dataset")
    bm25_idf_suparser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_suparser.add_argument("term", type=str, help="Term to get BM25 IDF score for")
    bm25_tf_suparser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given term")
    bm25_tf_suparser.add_argument("doc_id", type=int, help="Doc id")
    bm25_tf_suparser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_suparser.add_argument("K1", type=float, nargs="?", default=BM25_k1, help="Tunable BM25 K1 parameter")
    bm25_tf_suparser.add_argument("B", type=float, nargs="?", default=BM25_B, help="Tunable BM25 b parameter")
    bm25search_suparser = subparsers.add_parser("bm25search", help="Get BM25 search docs for a given term")
    bm25search_suparser.add_argument("query", type=str, help="Search query")
    bm25search_suparser.add_argument("limit", type=int, nargs="?", default=5, help="Tunable limit of docs to be returned")

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
        
        case "tfidf":
            Inverted_Index_Search.load()
            tf_idf = tfidf(args.term, args.doc_id, Inverted_Index_Search)
            print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            bm25_idf = bm25_idf_command(Inverted_Index_Search, args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25_idf:.2f}")
        
        case "bm25tf":
            Inverted_Index_Search.load()
            BM25_TF = Inverted_Index_Search.get_bm25_tf(args.doc_id, args.term, args.K1, args.B)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {BM25_TF:.2f}")

        case "bm25search":
            Inverted_Index_Search.load()
            dict_score = Inverted_Index_Search.bm25_search(args.query, args.limit)
            index = 1
            for k, v in dict_score.items():
                title = Inverted_Index_Search.docmap[k]["title"]
                print(f'{index}. ({k}) {title} - Score: {v:.2f}')
                index += 1
            
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