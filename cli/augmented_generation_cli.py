import argparse
from lib.hybrid_search import HybridSearch
from utils import file, load_file_json, K_VALUE, DEFAULT_LIMIT
from test_gemini import gemini_enhancer

def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser = subparsers.add_parser(
        "summarize", help="Summarizes the results returned by the Hybrid Search Class"
    )
    summarize_parser.add_argument("query", type=str, help="Search query")
    summarize_parser.add_argument("--limit", type=int, default=5, help="Limit of results")
    citations_parser = subparsers.add_parser(
        "citations", help="Answers with citations"
    )
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument("--limit", type=int, default=5, help="Limit of results")
    question_parser = subparsers.add_parser(
        "question", help="Answers a question"
    )
    question_parser.add_argument("query", type=str, help="Search query")
    question_parser.add_argument("--limit", type=int, default=5, help="Limit of results")

    args = parser.parse_args()

    query = args.query
    movies = load_file_json(file)
    hb = HybridSearch(movies["movies"])

    match args.command:
        case "rag":
            results = hb.rrf_search(query, K_VALUE, DEFAULT_LIMIT, "")
            rag_response = gemini_enhancer(query, "RAG", results)
            print("Search Results")
            for r in results:
                print("-", r[1]["document"]["title"])
            print()
            print("RAG Response:")
            print(rag_response)
        case "summarize":
            results = hb.rrf_search(query, K_VALUE, args.limit, "")
            summary = gemini_enhancer(query, "summarize", results)
            print("Search Results")
            for r in results:
                print("-", r[1]["document"]["title"])
            print()
            print("LLM Summary:")
            print(summary)
        case "citations":
            results = hb.rrf_search(query, K_VALUE, args.limit, "")
            citations = gemini_enhancer(query, "citations", results)
            print("Search Results")
            for r in results:
                print("-", r[1]["document"]["title"])
            print()
            print("LLM answer:")
            print(citations)
        case "question":
            results = hb.rrf_search(query, K_VALUE, args.limit, "")
            question = gemini_enhancer(query, "question", results)
            print("Search Results")
            for r in results:
                print("-", r[1]["document"]["title"])
            print()
            print("Answer:")
            print(question)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()