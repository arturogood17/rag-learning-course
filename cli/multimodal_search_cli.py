import argparse
from lib.multimodal_search import *

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    verify_image_suparser = subparsers.add_parser("verify_image_embedding", help="Verifies the embedding of an image")
    verify_image_suparser.add_argument(dest="image_path", type=str, help= "Path to the image to be embedded")
    image_search_suparser = subparsers.add_parser("image_search", help="Searches an image in the dataset")
    image_search_suparser.add_argument(dest="image_path", type=str, help= "Path to the image to be embedded")

    args = parser.parse_args()
    img_path = args.image_path.strip()

    match args.command:
        case "verify_image_embedding":
            verify_image_embedding(img_path)
        case "image_search":
            results = image_search_command(img_path)
            for index, r in enumerate(results, 1):
                print(f"{index}. {r["title"]} (similarity: {r["cosine_similarity"]:.3f})")
                print(f"         {r["description"][:100]}...")
                print()
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()