from .search_utils import DEFAULT_SEARCH_LIMIT, load_movies


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    movies = load_movies()

    matches = []
    for movie in movies:
        if query in movie["title"]:
            matches.append(movie)

    sorted_matches = sorted(matches, key=lambda match: match["id"])
    sliced_matches = sorted_matches[:5]

    return sliced_matches
