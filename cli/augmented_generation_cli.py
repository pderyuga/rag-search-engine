import argparse
from lib.augmented_generation import (
    rag_command,
    summarize_command,
    citations_command,
    question_command,
)
from lib.search_utils import DEFAULT_SEARCH_LIMIT


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser(
        "summarize", help="Generate multi-document summary"
    )
    summarize_parser.add_argument(
        "query", type=str, help="Search query for summarization"
    )
    summarize_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of documents to summarize",
    )

    citations_parser = subparsers.add_parser(
        "citations", help="Generate answer with citations"
    )
    citations_parser.add_argument(
        "query", type=str, help="Search query for answer with citations"
    )
    citations_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of documents to use",
    )

    question_parser = subparsers.add_parser(
        "question", help="Answer a question directly and concisely"
    )
    question_parser.add_argument("question", type=str, help="Question to answer")
    question_parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_SEARCH_LIMIT,
        help="Maximum number of documents to use",
    )

    args = parser.parse_args()

    match args.command:
        case "rag":
            rag_command(args.query)

        case "summarize":
            summarize_command(args.query, limit=args.limit)

        case "citations":
            citations_command(args.query, limit=args.limit)

        case "question":
            question_command(args.question, limit=args.limit)

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
