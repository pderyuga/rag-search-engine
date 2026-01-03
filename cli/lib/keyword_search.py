import os
import sys
import string
import math
import pickle
from typing import Set, TypedDict
from collections import Counter
from nltk.stem import PorterStemmer
from .search_utils import (
    DEFAULT_SEARCH_LIMIT,
    BM25_K1,
    BM25_B,
    load_movies,
    load_stopwords,
    CACHE_PATH,
    INDEX_PATH,
    DOCMAP_PATH,
    TF_PATH,
    DOC_LENGTHS_PATH,
)


def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT):
    idx = InvertedIndex()
    idx.load()

    matches = set()
    query_tokens = tokenize_text(query)
    for token in query_tokens:
        if token in idx.index:
            doc_ids = idx.index[token]
            matches.update(doc_ids)

    sorted_matches = sorted(list(matches))
    sliced_matches = sorted_matches[:limit]

    matching_movies: list[Movie] = []
    for doc_id in sliced_matches:
        if doc_id in idx.docmap:
            matching_movies.append(idx.docmap[doc_id])

    return matching_movies


def build_command():
    idx = InvertedIndex()
    idx.build()
    idx.save()


def tf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    idx.load()

    tf = 0
    if doc_id in idx.term_frequencies and term in idx.term_frequencies[doc_id]:
        tf = idx.term_frequencies[doc_id][term]

    return tf


def idf_command(term: str):
    idx = InvertedIndex()
    idx.load()

    idf = idx.get_idf(term)
    return idf


def tf_idf_command(doc_id: int, term: str):
    idx = InvertedIndex()
    idx.load()

    tf_idf = idx.get_tf_idf(doc_id, term)
    return tf_idf


def bm25_idf_command(term: str):
    idx = InvertedIndex()
    idx.load()

    bm25_idf = idx.get_bm25_idf(term)
    return bm25_idf


def bm25_tf_command(doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B):
    idx = InvertedIndex()
    idx.load()

    bm25_tf = idx.get_bm25_tf(doc_id, term, k1, b)
    return bm25_tf


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
        self.term_frequencies: dict[int, Counter] = {}
        self.doc_lengths: dict[int, int] = {}

    def __add_document(self, doc_id: int, text: str):
        # tokenize input text
        tokenized_text = tokenize_text(text)

        # initialize term_frequencies for this doc_id if it doesn't exist
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()

        # store doc length in doc_lengths
        self.doc_lengths[doc_id] = len(tokenized_text)

        # add each token to the index with the doc_id
        for token in tokenized_text:
            if token in self.index:
                self.index[token].add(doc_id)
            else:
                self.index[token] = set([doc_id])

            # Increment the term frequency
            self.term_frequencies[doc_id][token] += 1

    def __get_avg_doc_length(self):
        total_doc_count = len(self.docmap)
        if total_doc_count == 0:
            return 0.0
        total_doc_length = sum(self.doc_lengths.values())
        avg_doc_length = total_doc_length / total_doc_count
        return avg_doc_length

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
        with open(TF_PATH, "wb") as f:
            pickle.dump(self.term_frequencies, f)
        with open(DOC_LENGTHS_PATH, "wb") as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        try:
            with open(INDEX_PATH, "rb") as f:
                self.index = pickle.load(f)
            with open(DOCMAP_PATH, "rb") as f:
                self.docmap = pickle.load(f)
            with open(TF_PATH, "rb") as f:
                self.term_frequencies = pickle.load(f)
            with open(DOC_LENGTHS_PATH, "rb") as f:
                self.doc_lengths = pickle.load(f)
        except FileNotFoundError as e:
            print(f"Error: Cache files not found. Please run 'build' command first.")
            print(f"Details: {e}")
            sys.exit()
        except Exception as e:
            print(f"An unexpected error occurred while loading cache: {e}")
            sys.exit()

    def get_tf(self, doc_id: int, term: str):
        tokens = tokenize_text(term)
        if len(tokens) > 1:
            raise ValueError("Search term must be one word.")

        if len(tokens) == 0:
            return 0

        if tokens[0] in self.term_frequencies[doc_id]:
            return self.term_frequencies[doc_id][tokens[0]]
        else:
            return 0

    def get_idf(self, term: str):
        tokens = tokenize_text(term)

        if len(tokens) != 1:
            raise ValueError("Search term must be one word.")

        total_doc_count = len(self.docmap)
        if tokens[0] in self.index:
            term_match_doc_count = len(self.index[tokens[0]])
        else:
            term_match_doc_count = 0

        idf = math.log((total_doc_count + 1) / (term_match_doc_count + 1))
        return idf

    def get_tf_idf(self, doc_id: int, term: str):
        tokens = tokenize_text(term)

        if len(tokens) != 1:
            raise ValueError("Search term must be one word.")

        tf = self.get_tf(doc_id, term)
        idf = self.get_idf(term)
        tf_idf = tf * idf
        return tf_idf

    def get_bm25_idf(self, term: str):
        tokens = tokenize_text(term)

        if len(tokens) != 1:
            raise ValueError("Search term must be one word.")

        total_doc_count = len(self.docmap)
        if tokens[0] in self.index:
            term_match_doc_count = len(self.index[tokens[0]])
        else:
            term_match_doc_count = 0

        bm25_idf = math.log(
            (total_doc_count - term_match_doc_count + 0.5)
            / (term_match_doc_count + 0.5)
            + 1
        )
        return bm25_idf

    def get_bm25_tf(
        self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B
    ):
        doc_length = self.doc_lengths[doc_id]
        avg_doc_length = self.__get_avg_doc_length()

        # Length normalization factor
        length_norm = 1 - b + b * (doc_length / avg_doc_length)

        tf = self.get_tf(doc_id, term)

        bm25_tf = (tf * (k1 + 1)) / (tf + k1 * length_norm)
        return bm25_tf
