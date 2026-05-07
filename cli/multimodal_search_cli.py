import argparse
from lib.multimodal_search import *

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_image_suparser = subparsers.add_parser("verify_image_embedding", help="Verifies the embedding of an image")
    verify_image_suparser.add_argument(dest="image_path", type=str, help= "Path to the image to be embedded")

    args = parser.parse_args()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(args.image_path.strip())
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()