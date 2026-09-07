"""TF-IDF + cosine similarity over artist genre metadata (Signal 2).

Genres are atomic multi-word phrases ("indie folk", "chamber pop"), so
this deliberately treats each genre string as a single token instead of
letting TfidfVectorizer whitespace-tokenize it -- naive tokenizing would
wrongly conflate unrelated genres that happen to share a word (e.g. "pop").
"""

import json
import pickle

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config

logger = config.get_logger(__name__)

VECTORIZER_PATH = config.DATA_PROCESSED / "genre_tfidf_vectorizer.pkl"
MATRIX_PATH = config.DATA_PROCESSED / "genre_similarity_matrix.npz"


def _load_artist_genres() -> dict:
    path = config.SPOTIFY_RAW_DIR / "artist_genres.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def compute_similarity(
    artist_genres: dict,
) -> tuple[list[str], np.ndarray, TfidfVectorizer]:
    """Pure computation, no disk I/O -- shared by build() (real data) and
    evaluate.py's fixture-mode smoke test.

    Artists with no genres are excluded: they have no genre-similarity
    signal and fall back entirely to embeddings in recommend.py.
    """
    artist_ids, genre_lists = [], []
    skipped = 0
    for artist_id, info in artist_genres.items():
        genres = info.get("genres") or []
        if not genres:
            skipped += 1
            continue
        artist_ids.append(artist_id)
        genre_lists.append(genres)

    if skipped:
        logger.info("Excluded %d artists with no genre data", skipped)

    vectorizer = TfidfVectorizer(analyzer=lambda genres: genres)
    tfidf_matrix = vectorizer.fit_transform(genre_lists)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return artist_ids, similarity_matrix, vectorizer


def build(artist_genres: dict | None = None) -> tuple[list[str], np.ndarray]:
    """Fits the TF-IDF model, computes the pairwise similarity matrix, and
    persists both to data/processed/."""
    if artist_genres is None:
        artist_genres = _load_artist_genres()

    artist_ids, similarity_matrix, vectorizer = compute_similarity(artist_genres)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    with VECTORIZER_PATH.open("wb") as f:
        pickle.dump(vectorizer, f)
    np.savez(MATRIX_PATH, matrix=similarity_matrix, artist_ids=np.array(artist_ids))

    logger.info(
        "Indexed %d artists, %d genre terms", len(artist_ids), len(vectorizer.vocabulary_)
    )
    return artist_ids, similarity_matrix


def load_genre_model() -> tuple[list[str], np.ndarray]:
    data = np.load(MATRIX_PATH, allow_pickle=False)
    return list(data["artist_ids"]), data["matrix"]


def artist_similarity(
    artist_a: str, artist_b: str, artist_ids: list[str], sim_matrix: np.ndarray
) -> float | None:
    """Returns None if either artist is missing from the genre index --
    the genre-side cold-start fallback contract."""
    try:
        i, j = artist_ids.index(artist_a), artist_ids.index(artist_b)
    except ValueError:
        return None
    return float(sim_matrix[i, j])


def most_similar_artists(
    artist_id: str, artist_ids: list[str], sim_matrix: np.ndarray, top_n: int = 20
) -> list[tuple[str, float]]:
    if artist_id not in artist_ids:
        return []
    idx = artist_ids.index(artist_id)
    scores = sim_matrix[idx]
    ranked = sorted(
        (
            (artist_ids[i], float(scores[i]))
            for i in range(len(artist_ids))
            if i != idx
        ),
        key=lambda pair: pair[1],
        reverse=True,
    )
    return ranked[:top_n]


def main() -> None:
    artist_ids, sim_matrix = build()
    if artist_ids:
        sample = artist_ids[0]
        logger.info(
            "Nearest genre-neighbors for artist %s: %s",
            sample,
            most_similar_artists(sample, artist_ids, sim_matrix, top_n=5),
        )


if __name__ == "__main__":
    main()
