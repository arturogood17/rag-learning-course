import argparse
from lib.hybrid_search import *

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_subparser = subparsers.add_parser("normalize", help="Normalize the score between KS  and SM")
    normalize_subparser.add_argument(dest="list", nargs='*', type= float, help="List to be normalized")
    weighted_search_subparser = subparsers.add_parser("weighted-search", help="Weighted search with configurable KM and SM")
    weighted_search_subparser.add_argument("query", type=str, help= "Query to be searched")
    weighted_search_subparser.add_argument("--alpha", type=float, default=0.5, help= "Configurable alpha that affects KM and SM")
    weighted_search_subparser.add_argument("--limit", type=int, default=5, help= "Limit of results")
    rrf_search_subparser = subparsers.add_parser("rrf-search", help="Weighted search with configurable KM and SM")
    rrf_search_subparser.add_argument("query", type=str, help= "Query to be searched")
    rrf_search_subparser.add_argument("-k", type=int, default=60, help= "Configurable k parameter that affects weight of the ranks")
    rrf_search_subparser.add_argument("--limit", type=int, default=5, help= "Limit of results")
    rrf_search_subparser.add_argument("--enhance", type=str, choices=["spell"], help= "Query enhancement method")
    

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.list)
        case "weighted-search":
            weighted_search_command(args.query, args.alpha, args.limit)
        case "rrf-search":
            rrf_search_command(args.query, args.k, args.limit, args.enhance)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()