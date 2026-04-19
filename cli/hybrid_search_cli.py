import argparse
from lib.hybrid_search import normalize_command

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_subparser = subparsers.add_parser("normalize", help="Normalize the score between KS  and SM")
    normalize_subparser.add_argument(dest="list", nargs='*', type= float, help="List to be normalized")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(args.list)
        
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()