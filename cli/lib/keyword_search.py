import string
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()

    matches = []
    for movie in movies:
        preprocessed_query = preprocess_text(query)
        preprocessed_title = preprocess_text(movie["title"])
        if preprocessed_query in preprocessed_title:
            matches.append(movie)

    sorted_matches = sorted(matches, key=lambda match: match["id"])
    sliced_matches = sorted_matches[:5]

    return sliced_matches


def preprocess_text(text: str):
    text = text.lower()
    translation_table = str.maketrans("", "", string.punctuation)
    translated_text = text.translate(translation_table)
    return translated_text
