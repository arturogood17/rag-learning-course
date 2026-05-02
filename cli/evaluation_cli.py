import argparse
from utils import load_file_json, golden_dataset_path, file, K_VALUE
from lib.hybrid_search import HybridSearch

def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of results to evaluate (k for precision@k, recall@k)",
    )

    args = parser.parse_args()
    limit = args.limit

    golden_dataset = load_file_json(golden_dataset_path)
    movies = load_file_json(file)
    HB = HybridSearch(movies["movies"])
    retrieved = []
    for test in golden_dataset["test_cases"]:
        results = HB.rrf_search(test["query"], K_VALUE, limit, "")
        for r in results:
            retrieved.append(r[1]["document"]["title"])
        count = 0
        for r in retrieved:
            if r in test["relevant_docs"]:
                count += 1
        precision = count / len(retrieved)
        print(f"k={limit}")
        print(f"- Query: {test["query"]}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Retrieved: {", ".join(retrieved)}")
        print(f"  - Relevant: {", ".join(test["relevant_docs"])}")
        print()
        retrieved = []


if __name__ == "__main__":
    main()