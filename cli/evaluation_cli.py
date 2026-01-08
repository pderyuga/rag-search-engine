import argparse
from lib.evaluation import evaluate_command


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

    test_cases = evaluate_command(limit)

    print(f"k={limit}\n")
    for test_case in test_cases:
        print(f"- Query: {test_case["query"]}")
        print(f"  - Precision@{limit}: {test_case["precision"]:.4f}")
        print(f"  - Recall@{limit}: {test_case["recall"]:.4f}")
        print(f"  - Retrieved: {'. '.join(test_case["docs"])}")
        print(f"  - Relevant: {', '.join(test_case["relevant_docs"])}")
        print()


if __name__ == "__main__":
    main()
