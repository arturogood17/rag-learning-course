#!/usr/bin/env python3

import argparse
from lib.semantic_search import *

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    model_verify_suparser = subparsers.add_parser("verify", help="Verifies if the model is working")
    embedding_suparser = subparsers.add_parser("embed_text", help="Embeds the given text")
    embedding_suparser.add_argument("text", type=str, help= "Text to be embedded")
    verify_embedding_suparser = subparsers.add_parser("verify_embeddings", help="Verify if movies.json was embedded correctly")
    embedquery_suparser = subparsers.add_parser("embedquery", help="Embeds the given text")
    embedquery_suparser.add_argument("query", type=str, help= "Text to be embedded")
    search_suparser = subparsers.add_parser("search", help="Search semantically")
    search_suparser.add_argument("query", type=str, help= "Query")
    search_suparser.add_argument("--limit", type=int, nargs="?", default=5, help="Tunable limit of docs to be returned")


    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case "search":
            search(args.query, args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()