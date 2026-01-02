import os
import string
import pickle
from typing import Set, TypedDict
from nltk.stem import PorterStemmer
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    load_movies,
    load_stopwords,
    CACHE_PATH,
    INDEX_PATH,
    DOCMAP_PATH,
)


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


class Movie(TypedDict):
    id: int
    title: str
    description: str


class InvertedIndex:
    def __init__(self):
        self.index: dict[str, Set[int]] = {}
        self.docmap: dict[int, Movie] = {}

    def __add_document(self, doc_id: int, text: str):
        # tokenize input text
        tokenized_text = tokenize_text(text)
        # add each token to the index with the doc_id
        for token in set(tokenized_text):
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = set([doc_id])

    def get_documents(self, term: str):
        token = term.lower()
        # get a set of doc_id for a given token
        for current_token, doc_ids in self.index.items():
            if token == current_token:
                # return the doc ids as a list sorted in asc order
                return sorted(list(doc_ids))
        return []

    def build(self):
        movies = load_movies()
        # iterate over all the movies and add them to both the index and the docmap
        for movie in movies:
            self.__add_document(movie["id"], f"{movie["title"]} {movie["description"]}")
            self.docmap[movie["id"]] = movie

    def save(self):
        if not os.path.isdir(CACHE_PATH):
            os.makedirs(CACHE_PATH)

        with open(INDEX_PATH, "wb") as f:
            pickle.dump(self.index, f)
        with open(DOCMAP_PATH, "wb") as f:
            pickle.dump(self.docmap, f)
