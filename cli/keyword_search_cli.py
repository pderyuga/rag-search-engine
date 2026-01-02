#!/usr/bin/env python3

import argparse
from lib.keyword_search import search_command, build_command, tf_command, idf_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    subparsers.add_parser("build", help="Build the inverted index")

    tf_parser = subparsers.add_parser("tf", help="Check the frequency of a given term in a given document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term to get frequency for")

    idf_parser = subparsers.add_parser("idf", help="Get inverse document frequency for a given term")
    idf_parser.add_argument("term", type=str, help="Term to get IDF for")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")

            results = search_command(args.query)

            for index, match in enumerate(results):
                print(f"{index + 1}. {match["title"]}")

        case "build":
            print("Building inverted index...")
            build_command()
            print("Inverted index built successfully!")

        case "tf":
            tf = tf_command(args.doc_id, args.term)
            print(f"Term '{args.term}' appears {tf} time(s) in doc with id {args.doc_id}")

        case "idf":
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
