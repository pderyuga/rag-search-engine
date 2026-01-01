import string
from nltk.stem import PorterStemmer
from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies, load_stopwords


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()

    matches = []
    for movie in movies:
        query_tokens = tokenize_text(query)
        title_tokens = tokenize_text(movie["title"])
        is_item_present = has_matching_token(query_tokens, title_tokens)
        if is_item_present:
            matches.append(movie)

    sorted_matches = sorted(matches, key=lambda match: match["id"])
    sliced_matches = sorted_matches[:limit]

    return sliced_matches


def has_matching_token(query_tokens: list[str], title_tokens: list[str]):
    for query_token in query_tokens:
        for title_token in title_tokens:
            if query_token in title_token:
                return True
    return False


def preprocess_text(text: str):
    text = text.lower()
    translation_table = str.maketrans("", "", string.punctuation)
    translated_text = text.translate(translation_table)
    return translated_text


def tokenize_text(text: str):
    preprocessed_text = preprocess_text(text)
    tokens = preprocessed_text.split(" ")
    stopwords = load_stopwords()
    stemmer = PorterStemmer()

    valid_tokens = []
    for token in tokens:
        if token and not token in stopwords:
            stemmed_token = stemmer.stem(token)
            valid_tokens.append(stemmed_token)

    return valid_tokens
