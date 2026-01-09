import argparse
from lib.describe_image import (
    describe_image_command,
)
from lib.search_utils import DEFAULT_SEARCH_LIMIT


def main():
    parser = argparse.ArgumentParser(description="Describe Image CLI")
    parser.add_argument("--image", type=str, help="The path to an image file")
    parser.add_argument("--query", type=str, help="Text query")

    args = parser.parse_args()

    describe_image_command(args.image, args.query)


if __name__ == "__main__":
    main()
