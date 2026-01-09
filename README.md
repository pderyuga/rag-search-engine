# RAG Search Engine

A movie search engine built using Retrieval-Augmented Generation (RAG) concepts, implementing BM25 keyword search, semantic search with embeddings, hybrid search, multimodal image search, and AI-powered answer generation. This project is based on [Boot.dev's "Learn Retrieval Augmented Generation" course](https://www.boot.dev/courses/learn-retrieval-augmented-generation).

## Prerequisites

- Python 3.13
- `uv` package manager (for dependency management)
- Google Cloud Project with Vertex AI enabled (for AI-powered query enhancement and reranking)
- Dependencies:
  - `nltk==3.9.1` (for text processing)
  - `sentence-transformers>=5.2.0` (for semantic search embeddings)
  - `google-genai>=1.56.0` (for AI-powered query enhancement and results reranking)
  - `python-dotenv>=1.2.1` (for environment configuration)

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

### 5. Create Golden Dataset (Optional - for evaluation)

For testing and evaluating search quality, you'll need a golden dataset with test queries and known relevant documents.

Create `data/golden_dataset.json` and paste the following test cases:

```json
{
  "test_cases": [
    {
      "query": "cute british bear marmalade",
      "relevant_docs": ["Paddington"]
    },
    {
      "query": "talking teddy bear comedy",
      "relevant_docs": ["Ted", "Ted 2"]
    },
    {
      "query": "children's animated bear adventure",
      "relevant_docs": [
        "Brother Bear",
        "The Jungle Book",
        "The Many Adventures of Winnie the Pooh",
        "Yogi Bear",
        "The Care Bears Movie",
        "Care Bears Movie II: A New Generation",
        "Care Bears Nutcracker Suite",
        "The Little Polar Bear",
        "The Little Polar Bear 2: The Mysterious Island",
        "Open Season",
        "The Country Bears",
        "The Berenstain Bears' Christmas Tree",
        "Winnie the Pooh"
      ]
    },
    {
      "query": "friendship transformation magic with bears",
      "relevant_docs": [
        "Brother Bear",
        "The Care Bears Movie",
        "The Jungle Book"
      ]
    },
    {
      "query": "dinosaur park",
      "relevant_docs": ["Jurassic Park"]
    },
    {
      "query": "wizards and magic",
      "relevant_docs": [
        "Harry Potter and the Sorcerer's Stone",
        "Harry Potter and the Prisoner of Azkaban",
        "Harry Potter and the Goblet of Fire",
        "Harry Potter and the Order of the Phoenix",
        "Harry Potter and the Deathly Hallows: Part 1",
        "Harry Potter and the Deathly Hallows: Part 2",
        "The Sword in the Stone",
        "Oz the Great and Powerful",
        "The Lord of the Rings: The Fellowship of the Ring"
      ]
    },
    {
      "query": "superhero saves the world",
      "relevant_docs": [
        "The Incredibles",
        "Superman II",
        "Superman/Batman: Public Enemies",
        "Justice League: The Flashpoint Paradox",
        "Up, Up, and Away!",
        "Megamind",
        "Kick-Ass",
        "Sky High"
      ]
    },
    {
      "query": "zombie apocalypse",
      "relevant_docs": [
        "Shaun of the Dead",
        "Dance of the Dead",
        "The Return of the Living Dead",
        "Pride and Prejudice and Zombies",
        "I Am Legend",
        "Resident Evil: Apocalypse",
        "Colin",
        "Død snø"
      ]
    },
    {
      "query": "car racing",
      "relevant_docs": [
        "The Fast and the Furious",
        "Rush",
        "Need for Speed",
        "Talladega Nights: The Ballad of Ricky Bobby",
        "The Love Bug",
        "Cars",
        "Furious Seven"
      ]
    },
    {
      "query": "romantic comedy wedding",
      "relevant_docs": [
        "Runaway Bride",
        "27 Dresses",
        "Just Go with It",
        "The Wedding Planner",
        "Wedding Crashers",
        "The Accidental Husband",
        "You, Me and Dupree"
      ]
    }
  ]
}
```

This golden dataset is used by the evaluation system to measure Precision@k, Recall@k, and F1 scores across various query types.

### 6. Configure Gemini/Vertex AI (Optional - for AI-powered features)

For query enhancement and results reranking features, configure your Google Cloud credentials:

Copy the template file and configure your credentials:

```bash
cp .env.template .env
```

Then edit `.env` and replace the placeholder values with your actual Google Cloud configuration:

```bash
GEMINI_PROJECT=your-gcp-project-id
GEMINI_LOCATION=your-region (e.g., us-central1)
```

**Note:** These features require:

- A Google Cloud Project with Vertex AI API enabled
- Appropriate authentication (gcloud CLI configured or service account)
- The AI features (`--enhance` and `--rerank-method` flags) are optional and the search engine works without them

### 7. Build the Inverted Index

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

### BM25 IDF Command

Get the BM25 inverse document frequency for a term:

```bash
python cli/keyword_search_cli.py bm25idf <term>
```

Example:

```bash
python cli/keyword_search_cli.py bm25idf "princess"
```

### BM25 TF Command

Get the BM25 term frequency for a term in a specific document. Optionally specify k1 and b tuning parameters:

```bash
python cli/keyword_search_cli.py bm25tf <document_id> <term> [k1] [b]
```

Example:

```bash
python cli/keyword_search_cli.py bm25tf 1 "princess"
python cli/keyword_search_cli.py bm25tf 1 "princess" 1.5 0.75
```

### BM25 Search Command

Search for movies using the full BM25 ranking algorithm (more sophisticated than basic search):

```bash
python cli/keyword_search_cli.py bm25search "your search query"
```

Example:

```bash
python cli/keyword_search_cli.py bm25search "space adventure"
```

This returns movies ranked by BM25 scores, which considers term frequency, document frequency, and document length normalization.

## Semantic Search

The project includes a semantic search CLI that uses sentence transformer embeddings for meaning-based search (as opposed to keyword matching).

### Verify Embedding Model

Verify that the sentence transformer embedding model is properly loaded:

```bash
python cli/semantic_search_cli.py verify
```

This command will download the embedding model on first run and verify it's working correctly.

### Embed Text Command

Generate an embedding vector for any text and view its dimensions:

```bash
python cli/semantic_search_cli.py embed_text "your text here"
```

Example:

```bash
python cli/semantic_search_cli.py embed_text "space adventure movie"
```

This displays the first 3 dimensions and total dimensionality of the embedding vector.

### Verify Embeddings Command

Build or verify embeddings for the entire movie dataset:

```bash
python cli/semantic_search_cli.py verify_embeddings
```

This loads existing embeddings from cache or creates new ones if needed, then displays the embedding matrix shape.

### Embed Query Command

Generate an embedding specifically for a search query:

```bash
python cli/semantic_search_cli.py embedquery "your query"
```

Example:

```bash
python cli/semantic_search_cli.py embedquery "romantic comedy"
```

### Semantic Search Command

Search for movies using semantic similarity (the main semantic search feature):

```bash
python cli/semantic_search_cli.py search "your query" [--limit N]
```

Example:

```bash
python cli/semantic_search_cli.py search "space adventure"
python cli/semantic_search_cli.py search "love story" --limit 5
```

This uses cosine similarity between query and movie embeddings to find semantically similar movies, even if they don't share exact keywords.

### Chunk Text Command

Split text into fixed-size word-based chunks with optional overlap:

```bash
python cli/semantic_search_cli.py chunk "your text" [--chunk-size N] [--overlap N]
```

Example:

```bash
python cli/semantic_search_cli.py chunk "This is a long text that needs chunking" --chunk-size 5 --overlap 2
```

Useful for processing long documents that exceed embedding model limits.

### Semantic Chunk Command

Split text into sentence-based semantic chunks with optional overlap:

```bash
python cli/semantic_search_cli.py semantic_chunk "your text" [--max-chunk-size N] [--overlap N]
```

Example:

```bash
python cli/semantic_search_cli.py semantic_chunk "First sentence. Second sentence. Third sentence." --max-chunk-size 2 --overlap 1
```

This respects sentence boundaries for more meaningful chunks.

### Embed Chunks Command

Generate embeddings for chunked movie descriptions:

```bash
python cli/semantic_search_cli.py embed_chunks
```

This preprocesses the movie dataset by:

- Splitting each movie description into semantic chunks
- Generating embeddings for each chunk
- Saving chunk embeddings and metadata to cache

Useful for handling long documents that exceed embedding model limits. Run this once before using chunked search.

### Search Chunked Command

Search using chunked document embeddings:

```bash
python cli/semantic_search_cli.py search_chunked "your query" [--limit N]
```

Example:

```bash
python cli/semantic_search_cli.py search_chunked "romantic comedy" --limit 10
```

This searches at the chunk level and aggregates results by movie, keeping the highest scoring chunk for each movie. Better for finding specific details in long descriptions.

## Hybrid Search

Hybrid search combines both BM25 keyword search and semantic search to leverage the strengths of both approaches. It normalizes scores from each method and combines them for better results.

### Normalize Scores Command

Normalize a list of scores to the 0-1 range (useful for understanding score normalization):

```bash
python cli/hybrid_search_cli.py normalize <score1> <score2> [score3...]
```

Example:

```bash
python cli/hybrid_search_cli.py normalize 0.5 2.3 1.2 0.5 0.1
```

### Weighted Hybrid Search Command

Search using a weighted combination of BM25 and semantic scores:

```bash
python cli/hybrid_search_cli.py weighted-search "query" [--alpha N] [--limit N]
```

Example:

```bash
python cli/hybrid_search_cli.py weighted-search "space adventure"
python cli/hybrid_search_cli.py weighted-search "romantic comedy" --alpha 0.7 --limit 5
```

The `alpha` parameter controls the weight distribution:

- `alpha = 0.0`: Pure semantic search
- `alpha = 0.5`: Balanced (default)
- `alpha = 1.0`: Pure BM25 keyword search

### RRF Hybrid Search Command

Search using Reciprocal Rank Fusion (alternative hybrid approach):

```bash
python cli/hybrid_search_cli.py rrf-search "query" [--k N] [--limit N] [--enhance METHOD] [--rerank-method METHOD]
```

Example:

```bash
python cli/hybrid_search_cli.py rrf-search "space adventure"
python cli/hybrid_search_cli.py rrf-search "action thriller" --k 60 --limit 10
```

RRF combines rankings from both search methods without requiring score normalization. The `k` parameter (default=60) controls how much weight is given to lower-ranked results.

#### AI-Powered Query Enhancement (Optional)

Use the `--enhance` flag to improve search queries with Gemini AI:

```bash
python cli/hybrid_search_cli.py rrf-search "query" --enhance METHOD
```

Enhancement methods:

- **spell**: Fix spelling errors in the query
- **rewrite**: Rewrite the query to be more specific and searchable
- **expand**: Add synonyms and related terms to the query

Examples:

```bash
# Fix spelling errors
python cli/hybrid_search_cli.py rrf-search "romntic commedy" --enhance spell

# Rewrite vague queries to be more specific
python cli/hybrid_search_cli.py rrf-search "that bear movie" --enhance rewrite

# Expand with related terms
python cli/hybrid_search_cli.py rrf-search "scary movie" --enhance expand
```

#### AI-Powered Results Reranking (Optional)

Use the `--rerank-method` flag to reorder results using AI for better relevance:

```bash
python cli/hybrid_search_cli.py rrf-search "query" --rerank-method METHOD
```

Reranking methods:

- **individual**: LLM scores each result individually (0-10 rating)
- **batch**: LLM ranks all results at once (more efficient)
- **cross_encoder**: Uses CrossEncoder model for relevance scoring (faster, no API calls)

Examples:

```bash
# Individual scoring with Gemini
python cli/hybrid_search_cli.py rrf-search "space adventure" --rerank-method individual

# Batch reranking with Gemini
python cli/hybrid_search_cli.py rrf-search "romantic comedy" --rerank-method batch --limit 10

# CrossEncoder reranking (local, no API)
python cli/hybrid_search_cli.py rrf-search "action thriller" --rerank-method cross_encoder
```

You can combine both enhancement and reranking:

```bash
python cli/hybrid_search_cli.py rrf-search "scary bear movie" --enhance rewrite --rerank-method batch --limit 5
```

#### AI-Powered Results Evaluation (Optional)

Use the `--evaluate` flag to get AI-powered quality assessment of search results:

```bash
python cli/hybrid_search_cli.py rrf-search "query" --evaluate
```

This uses Gemini AI to judge each result's relevance on a 0-3 scale:

- **3**: Highly relevant
- **2**: Relevant
- **1**: Marginally relevant
- **0**: Not relevant

Example:

```bash
python cli/hybrid_search_cli.py rrf-search "romantic comedy" --limit 10 --evaluate
```

You can combine evaluation with enhancement and reranking:

```bash
python cli/hybrid_search_cli.py rrf-search "space adventure" --enhance rewrite --rerank-method batch --evaluate
```

## Search Evaluation

The evaluation system helps measure and improve search quality using standardized information retrieval metrics.

### Evaluation Command

Evaluate search performance using a golden dataset:

```bash
python cli/evaluation_cli.py [--limit N]
```

The `--limit` parameter sets the k value for Precision@k and Recall@k calculations (default=5).

Example:

```bash
python cli/evaluation_cli.py
python cli/evaluation_cli.py --limit 10
```

This command:

1. Runs RRF search on test queries from `data/golden_dataset.json`
2. Compares results against known relevant documents
3. Calculates metrics for each query:
   - **Precision@k**: What fraction of retrieved documents are relevant?
   - **Recall@k**: What fraction of relevant documents were retrieved?
   - **F1 Score**: Harmonic mean of precision and recall

Example output:

```
k=5

- Query: space adventure movies
  - Precision@5: 0.8000
  - Recall@5: 0.6667
  - F1 Score: 0.7273
  - Retrieved: Star Wars. The Empire Strikes Back. Alien. Interstellar. Blade Runner
  - Relevant: Star Wars, The Empire Strikes Back, Alien, Interstellar, Gravity, The Martian
```

### Understanding the Metrics

- **Precision@k**: Of the top k results returned, what percentage are actually relevant? Higher is better. Perfect score = 1.0.
- **Recall@k**: Of all the relevant documents in the dataset, what percentage did we find in the top k results? Higher is better. Perfect score = 1.0.

- **F1 Score**: Balances precision and recall into a single metric. Useful when you care about both missing relevant documents (low recall) and including irrelevant ones (low precision).

These metrics help you:

- Compare different search algorithms
- Tune parameters (like alpha, k, enhancement methods)
- Identify queries where search performs poorly
- Track improvements over time

## Multimodal Search

Multimodal search uses the CLIP (Contrastive Language-Image Pre-Training) model to search for movies using images instead of text queries. This enables visual similarity search.

### Verify Image Embedding Command

Verify that the CLIP model can generate embeddings from an image:

```bash
python cli/multimodal_search_cli.py verify_image_embedding "path/to/image.jpg"
```

Example:

```bash
python cli/multimodal_search_cli.py verify_image_embedding "data/paddington.jpeg"
```

This loads the image, generates an embedding, and displays the embedding dimensions.

### Image Search Command

Search for movies using an image:

```bash
python cli/multimodal_search_cli.py image_search "path/to/image.jpg"
```

Example:

```bash
python cli/multimodal_search_cli.py image_search "data/paddington.jpeg"
```

This finds movies whose descriptions are semantically similar to the content of the image. The CLIP model understands both images and text in a shared embedding space, enabling cross-modal search.

**Use cases:**

- Find movies similar to a movie poster or screenshot
- Search using concept images (e.g., a space scene to find sci-fi movies)
- Visual similarity search without needing text descriptions

## Image Description

Use Gemini AI's vision capabilities to describe images or answer questions about them.

### Describe Image Command

Describe an image or answer a question about it:

```bash
python cli/describe_image_cli.py --image "path/to/image.jpg" --query "your question"
```

Examples:

```bash
# General description
python cli/describe_image_cli.py --image "data/paddington.jpeg" --query "Describe this image"

# Specific question
python cli/describe_image_cli.py --image "data/movie_poster.jpg" --query "What genre is this movie?"
```

This uses Gemini's multimodal capabilities to analyze images and provide natural language responses.

## Retrieval-Augmented Generation (RAG)

RAG combines search with AI generation to provide accurate, context-aware answers grounded in your document collection. Instead of relying solely on the LLM's training data, RAG retrieves relevant documents first and uses them to generate informed responses.

### RAG Command

Basic RAG: search for relevant movies and generate an answer:

```bash
python cli/augmented_generation_cli.py rag "your query"
```

Example:

```bash
python cli/augmented_generation_cli.py rag "What are some good space adventure movies?"
```

This performs a search, retrieves relevant movie descriptions, and generates a natural language answer based on the retrieved content.

### Summarize Command

Generate a multi-document summary from search results:

```bash
python cli/augmented_generation_cli.py summarize "query" [--limit N]
```

Example:

```bash
python cli/augmented_generation_cli.py summarize "romantic comedies" --limit 5
```

Retrieves multiple relevant documents and generates a comprehensive summary synthesizing information from all of them.

### Citations Command

Generate an answer with source citations:

```bash
python cli/augmented_generation_cli.py citations "query" [--limit N]
```

Example:

```bash
python cli/augmented_generation_cli.py citations "movies about artificial intelligence" --limit 5
```

Provides an answer along with explicit citations showing which movies/documents the information came from. Useful for transparency and fact-checking.

### Question Command

Answer a specific question directly and concisely:

```bash
python cli/augmented_generation_cli.py question "your question" [--limit N]
```

Example:

```bash
python cli/augmented_generation_cli.py question "Who directed The Godfather?" --limit 3
```

Optimized for direct, factual questions where you need a concise answer rather than a detailed explanation.

**Why RAG?**

- **Accuracy**: Answers grounded in actual documents, not hallucinated
- **Current**: Works with your latest data, not just LLM training cutoff
- **Transparent**: Can cite sources and show retrieved documents
- **Controllable**: Uses your curated document collection

## Project Structure

```
rag-search-engine/
├── cache/                        # Cache directory (created after build)
│   ├── index.pkl                 # Pickled inverted index
│   ├── docmap.pkl                # Pickled document mapping
│   ├── tf.pkl                    # Pickled term frequencies
│   ├── doc_lengths.pkl           # Pickled document lengths
│   ├── movie_embeddings.npy      # Cached movie embeddings for semantic search
│   ├── chunk_embeddings.npy      # Cached chunk embeddings for chunked search
│   └── chunk_metadata.json       # Metadata mapping chunks to movies
├── cli/
│   ├── keyword_search_cli.py      # Keyword search CLI entry point
│   ├── semantic_search_cli.py     # Semantic search CLI entry point
│   ├── hybrid_search_cli.py       # Hybrid search CLI entry point
│   ├── multimodal_search_cli.py   # Multimodal image search CLI entry point
│   ├── describe_image_cli.py      # Image description CLI entry point
│   ├── augmented_generation_cli.py # RAG (Retrieval-Augmented Generation) CLI entry point
│   ├── evaluation_cli.py          # Search evaluation CLI entry point
│   ├── test_gemini.py             # Test script for Gemini AI integration
│   └── lib/
│       ├── keyword_search.py      # Keyword search implementation
│       ├── semantic_search.py     # Semantic search implementation
│       ├── hybrid_search.py       # Hybrid search implementation
│       ├── multimodal_search.py   # Multimodal CLIP-based image search
│       ├── describe_image.py      # Gemini vision for image description
│       ├── augmented_generation.py # RAG implementation (search + generation)
│       ├── query_enhancement.py   # AI-powered query enhancement (spell, rewrite, expand)
│       ├── results_reranking.py   # AI-powered results reranking
│       ├── evaluation.py          # Search evaluation metrics and LLM judging
│       └── search_utils.py        # Shared utility functions
├── data/
│   ├── movies.json               # Movie dataset (download required)
│   ├── stopwords.txt             # Stop words for text processing
│   └── golden_dataset.json       # Test queries with known relevant documents (optional)
├── .env                          # Environment variables (Gemini/Vertex AI config)
├── .env.template                 # Template for environment variables configuration
├── .gitignore                    # Git ignore rules
├── pyproject.toml                # Project configuration
├── uv.lock                       # Dependency lock file
└── README.md                     # This file
```

**Note:** The `cache/` directory is automatically created when you run the `build` command (for keyword search) or when you first run semantic/hybrid search commands. Chunked search creates additional cache files (`chunk_embeddings.npy` and `chunk_metadata.json`) when you run `embed_chunks`. The sentence transformer model itself is downloaded by the library and cached separately on first use.

## Notes

This documentation will be updated as the project evolves and more features are added throughout the Boot.dev RAG course.
