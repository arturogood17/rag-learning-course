#!/usr/bin/env python3

import argparse
from lib.semantic_search import *

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    model_verify_suparser = subparsers.add_parser("verify", help="Verifies if the model is working")
    embedding_suparser = subparsers.add_parser("embed_text", help="Embeds the given text")
    embedding_suparser.add_argument("text", type=str, help= "Text to be embedded")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()
        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()