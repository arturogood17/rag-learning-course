import mimetypes, os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import argparse


def main():
    parser = argparse.ArgumentParser(description="Multimodal RAG")
    parser.add_argument("--image", type=str, help="Path to image")
    parser.add_argument("--query", type=str, help="Query to search")

    args = parser.parse_args()
    path_to_image = args.image
    query = args.query

    response = multimodal_query_rewriting(path_to_image, query)
    print(f"Rewritten query: {response.text.strip()}")
    if response.usage_metadata is not None:
        print(f"Total tokens:    {response.usage_metadata.total_token_count}")


def multimodal_query_rewriting(image: str, query: str) -> any:
    mime = mimetypes.guess_type(image)
    mime = mime or 'image/jpeg'
    with open(image, 'rb') as f:
        image_content = f.read()
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY environment variable not set")
    prompt = """Given the included image and text query, rewrite the text query to improve search results from a movie database. Make sure to:
- Synthesize visual and textual information
- Focus on movie-specific details (actors, scenes, style, etc.)
- Return only the rewritten query, without any additional commentary"""

    client = genai.Client(api_key= api_key)
    response = client.models.generate_content(model= 'gemma-4-31b-it', contents= [prompt,
                                                                                 types.Part.from_bytes(data= image_content, mime_type= mime[0]),
                                                                                 query.strip()])
    return response

if __name__ == "__main__":
    main()