"""Trains Word2Vec-style track embeddings on the Million Playlist Dataset
(Signal 1). One playlist = one "sentence" of track-ID tokens.

Requires the MPD already downloaded into data/raw/mpd/ (see fetch.py's
module docstring for the download URL). Run directly:

    python src/train_embeddings.py --num-playlists 5000
"""

import argparse
import json
from collections.abc import Iterable

from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors

import config
import fetch

logger = config.get_logger(__name__)

TRACK_URI_PREFIX = "spotify:track:"
ARTIST_URI_PREFIX = "spotify:artist:"

# Word2Vec hyperparameters.
VECTOR_SIZE = 100
# Wide window: playlist track *order* is a weak signal, so this approximates
# "co-occurs anywhere in the playlist" rather than local sequence.
WINDOW = 10
# Literal cold-start boundary: a track needs >=5 playlist appearances in the
# training subset to get a vector. Below this, recommend.py falls back to
# genre-only scoring (see is_in_vocabulary()).
MIN_COUNT = 5
EPOCHS = 10


def _track_id(uri: str) -> str:
    return uri.removeprefix(TRACK_URI_PREFIX)


def _artist_id(uri: str) -> str:
    return uri.removeprefix(ARTIST_URI_PREFIX)


def build_corpus_from_slices(
    slices: Iterable[dict], num_playlists: int | None = None
) -> tuple[list[list[str]], dict]:
    """Pure corpus-building logic shared by load_corpus() (real MPD) and
    evaluate.py's fixture-mode smoke test.

    Returns (sentences, track_catalog). track_catalog maps every track ID
    seen (not just embedding-vocabulary survivors) to
    {name, artist_id, artist_name, occurrence_count} -- this is what lets
    recommend.py actually exercise the cold-start fallback instead of only
    ever seeing tracks that already have embeddings.
    """
    sentences: list[list[str]] = []
    catalog: dict[str, dict] = {}

    for slice_data in slices:
        for playlist in slice_data.get("playlists", []):
            tracks = sorted(playlist.get("tracks", []), key=lambda t: t["pos"])
            token_ids = [_track_id(t["track_uri"]) for t in tracks]
            for token_id, track in zip(token_ids, tracks):
                entry = catalog.setdefault(
                    token_id,
                    {
                        "name": track["track_name"],
                        "artist_id": _artist_id(track["artist_uri"]),
                        "artist_name": track["artist_name"],
                        "occurrence_count": 0,
                    },
                )
                entry["occurrence_count"] += 1
            if len(token_ids) >= 2:
                sentences.append(token_ids)
            if num_playlists and len(sentences) >= num_playlists:
                return sentences, catalog

    return sentences, catalog


def load_corpus(num_playlists: int) -> tuple[list[list[str]], dict]:
    paths = fetch.load_mpd_slice_paths()
    return build_corpus_from_slices(fetch.iter_mpd_slices(paths), num_playlists)


def train_word2vec(
    sentences: list[list[str]],
    vector_size: int = VECTOR_SIZE,
    window: int = WINDOW,
    min_count: int = MIN_COUNT,
    epochs: int = EPOCHS,
) -> Word2Vec:
    return Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=1,  # skip-gram: better than CBOW for the long tail of infrequent tracks
        epochs=epochs,
        seed=config.RANDOM_SEED,
    )


def is_in_vocabulary(kv: KeyedVectors, track_id: str) -> bool:
    """Cold-start contract: every caller MUST check this before calling
    .similarity()/.most_similar() on a track ID -- missing tracks fall back
    to genre-only scoring, never a crash."""
    return track_id in kv.key_to_index


def train(num_playlists: int = 5000) -> None:
    sentences, catalog = load_corpus(num_playlists)
    logger.info("Loaded %d playlists, %d unique tracks", len(sentences), len(catalog))

    model = train_word2vec(sentences)

    config.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    model.save(str(config.DATA_PROCESSED / "word2vec.model"))
    model.wv.save(str(config.DATA_PROCESSED / "track_vectors.kv"))

    with (config.DATA_PROCESSED / "track_catalog.json").open("w", encoding="utf-8") as f:
        json.dump(catalog, f, indent=2)

    metadata = {
        "num_playlists": len(sentences),
        "vocab_size": len(model.wv.key_to_index),
        "catalog_size": len(catalog),
        "hyperparameters": {
            "vector_size": VECTOR_SIZE,
            "window": WINDOW,
            "min_count": MIN_COUNT,
            "sg": 1,
            "epochs": EPOCHS,
            "seed": config.RANDOM_SEED,
        },
        "source_slices": [p.name for p in fetch.load_mpd_slice_paths()],
    }
    with (config.DATA_PROCESSED / "embedding_metadata.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Saved model (%d in-vocabulary tracks of %d total seen)",
        len(model.wv.key_to_index),
        len(catalog),
    )


def load_track_vectors() -> KeyedVectors:
    return KeyedVectors.load(str(config.DATA_PROCESSED / "track_vectors.kv"))


def load_track_catalog() -> dict:
    with (config.DATA_PROCESSED / "track_catalog.json").open(
        "r", encoding="utf-8"
    ) as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-playlists", type=int, default=5000)
    args = parser.parse_args()
    train(num_playlists=args.num_playlists)


if __name__ == "__main__":
    main()
