# RAG Search Engine

A movie search engine built using Retrieval-Augmented Generation (RAG) concepts, implementing BM25 keyword search with Python. This project is based on [Boot.dev's "Learn Retrieval Augmented Generation" course](https://www.boot.dev/courses/learn-retrieval-augmented-generation).

## Prerequisites

- Python 3.13
- `uv` package manager (for dependency management)

## Setup Instructions

### 1. Install Dependencies

```bash
uv sync
```

This will automatically create the virtual environment (`.venv`) and install all required dependencies including `nltk==3.9.1`.

### 2. Activate Virtual Environment

```bash
source .venv/bin/activate
```

### 3. Download Movie Dataset

Download the movie dataset from the following URL and save it to `data/movies.json`:

```bash
curl -o data/movies.json https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json
```

Or manually download from: https://storage.googleapis.com/qvault-webapp-dynamic-assets/course_assets/course-rag-movies.json

### 4. Create Stop Words File

Create a file at `data/stopwords.txt` and paste the following stop words list:

```
a
about
above
after
again
against
ain
all
am
an
and
any
are
aren
aren't
as
at
be
because
been
before
being
below
between
both
but
by
can
couldn
couldn't
d
did
didn
didn't
do
does
doesn
doesn't
doing
don
don't
down
during
each
few
for
from
further
had
hadn
hadn't
has
hasn
hasn't
have
haven
haven't
having
he
he'd
he'll
he's
her
here
hers
herself
him
himself
his
how
i
i'd
i'll
i'm
i've
if
in
into
is
isn
isn't
it
it'd
it'll
it's
its
itself
just
ll
m
ma
me
mightn
mightn't
more
most
mustn
mustn't
my
myself
needn
needn't
no
nor
not
now
o
of
off
on
once
only
or
other
our
ours
ourselves
out
over
own
re
s
same
shan
shan't
she
she'd
she'll
she's
should
should've
shouldn
shouldn't
so
some
such
t
than
that
that'll
the
their
theirs
them
themselves
then
there
these
they
they'd
they'll
they're
they've
this
those
through
to
too
under
until
up
ve
very
was
wasn
wasn't
we
we'd
we'll
we're
we've
were
weren
weren't
what
when
where
which
while
who
whom
why
will
with
won
won't
wouldn
wouldn't
y
you
you'd
you'll
you're
you've
your
yours
yourself
yourselves
```

### 5. Build the Inverted Index

Before you can search, you need to build the inverted index from the movie data:

```bash
python cli/keyword_search_cli.py build
```

This will process all movies and create cached index files in the `.cache/` directory. You only need to run this once (or whenever you update the movie dataset).

## Usage

The CLI provides several commands for searching and analyzing the movie dataset:

### Build Command

Build or rebuild the inverted index:

```bash
python cli/keyword_search_cli.py build
```

### Search Command

Search for movies using keyword matching:

```bash
python cli/keyword_search_cli.py search "your search query"
```

Example:

```bash
python cli/keyword_search_cli.py search "space adventure"
```

This will return a ranked list of movies matching your query.

### Term Frequency (TF) Command

Get the frequency of a specific term in a specific document:

```bash
python cli/keyword_search_cli.py tf <document_id> <term>
```

Example:

```bash
python cli/keyword_search_cli.py tf 1 "princess"
```

### Inverse Document Frequency (IDF) Command

Get the inverse document frequency for a term across all documents:

```bash
python cli/keyword_search_cli.py idf <term>
```

Example:

```bash
python cli/keyword_search_cli.py idf "princess"
```

### TF-IDF Command

Get the TF-IDF score for a term in a specific document:

```bash
python cli/keyword_search_cli.py tfidf <document_id> <term>
```

Example:

```bash
python cli/keyword_search_cli.py tfidf 1 "princess"
```

## Project Structure

```
rag-search-engine/
├── .cache/                       # Cache directory (created after build)
│   ├── index.pkl                 # Pickled inverted index
│   ├── docmap.pkl                # Pickled document mapping
│   └── tf.pkl                    # Pickled term frequencies
├── cli/
│   ├── keyword_search_cli.py    # Main CLI entry point
│   └── lib/
│       ├── keyword_search.py     # Search implementation
│       └── search_utils.py       # Utility functions
├── data/
│   ├── movies.json               # Movie dataset (download required)
│   └── stopwords.txt             # Stop words for text processing
├── .gitignore                    # Git ignore rules
├── pyproject.toml                # Project configuration
├── uv.lock                       # Dependency lock file
└── README.md                     # This file
```

**Note:** The `.cache/` directory is automatically created when you run the `build` command and contains the processed index data for fast lookups.

## Notes

This documentation will be updated as the project evolves and more features are added throughout the Boot.dev RAG course.
