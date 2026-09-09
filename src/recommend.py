"""Blends Signal 1 (embeddings) + Signal 2 (genre similarity) into ranked,
explainable recommendations, with an LLM natural-language query layer that
extracts genre + time-window intent only (CLAUDE.md's query-layer scope).

evaluate.py calls score_candidates()/explain() directly and never goes
through extract_intent() or the CLI confirmation flow below -- that keeps
evaluation deterministic, free of LLM calls, and safe to run in CI.

Run directly:

    python src/recommend.py "upbeat songs like my recent listening" --no-llm
    python src/recommend.py --interactive
"""

import argparse
import difflib
from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from gensim.models.keyedvectors import KeyedVectors
from google import genai
from google.genai import types as genai_types

import config
import fetch
import genre_similarity
import train_embeddings

logger = config.get_logger(__name__)

TIME_WINDOW_SOURCES = {
    "recent": ["recently_played", "top_tracks_short_term"],
    "medium_term": ["top_tracks_medium_term"],
    "long_term": ["top_tracks_long_term"],
    "all_time": [
        "recently_played",
        "top_tracks_short_term",
        "top_tracks_medium_term",
        "top_tracks_long_term",
        "saved_tracks",
    ],
}

CANDIDATE_TOPN_EMBEDDING = 200
CANDIDATE_TOPN_GENRE_ARTISTS = 20

# Free-tier eligible via a Google AI Studio key; check aistudio.google.com
# for whichever flash model is currently free/current if this drifts.
INTENT_MODEL = "gemini-2.5-flash"
INTENT_TOOL = genai_types.FunctionDeclaration(
    name="record_intent",
    description=(
        "Records the extracted genre and time-window intent from the "
        "user's natural-language song recommendation request."
    ),
    parameters=genai_types.Schema(
        type=genai_types.Type.OBJECT,
        properties={
            "genre": genai_types.Schema(
                type=genai_types.Type.STRING,
                nullable=True,
                description="A single genre mentioned or implied by the query, or null if none is specified.",
            ),
            "time_window": genai_types.Schema(
                type=genai_types.Type.STRING,
                nullable=True,
                enum=["recent", "medium_term", "long_term", "all_time"],
                description="Which listening-history window the user means.",
            ),
            "needs_clarification": genai_types.Schema(type=genai_types.Type.BOOLEAN),
            "clarifying_question": genai_types.Schema(
                type=genai_types.Type.STRING,
                nullable=True,
                description="A specific question to ask if needs_clarification is true, else null.",
            ),
        },
        required=["genre", "time_window", "needs_clarification", "clarifying_question"],
    ),
)
INTENT_SYSTEM_PROMPT = """You extract exactly two pieces of intent from a user's request for song \
recommendations: a genre, and a time window (one of: recent, medium_term, long_term, \
all_time). Never infer mood, tempo, or energy -- that data isn't available for this system; \
if the user asks for something mood-based, set needs_clarification=true and ask them to \
rephrase in terms of genre or time period instead. If genre or time window isn't reasonably \
clear from the query, set needs_clarification=true with a specific clarifying_question \
rather than guessing."""


@dataclass
class SeedProfile:
    track_ids: set
    artist_ids: set


@dataclass
class ScoredTrack:
    track_id: str
    name: str
    artist_id: str
    artist_name: str
    signal_embedding: float | None
    signal_genre: float | None
    final_score: float
    why: str


@dataclass
class IntentResult:
    genre: str | None
    time_window: str | None
    needs_clarification: bool
    clarifying_question: str | None


# --- Seed profile / candidate generation ---------------------------------


def build_seed_profile(time_window: str = "all_time") -> SeedProfile:
    sources = TIME_WINDOW_SOURCES.get(time_window, TIME_WINDOW_SOURCES["all_time"])
    track_ids, artist_ids = set(), set()
    for source in sources:
        for track in fetch.load_tracks(source):
            if track.get("id"):
                track_ids.add(track["id"])
            for artist in track.get("artists", []):
                if artist.get("id"):
                    artist_ids.add(artist["id"])
    return SeedProfile(track_ids=track_ids, artist_ids=artist_ids)


def build_artist_track_index(catalog: dict) -> dict:
    index = defaultdict(list)
    for track_id, info in catalog.items():
        index[info["artist_id"]].append(track_id)
    return index


def generate_candidates(
    seed_track_ids: set,
    seed_artist_ids: set,
    kv: KeyedVectors,
    artist_ids: list,
    sim_matrix: np.ndarray,
    artist_track_index: dict,
) -> set:
    """Generate-then-rank candidate generation: union of embedding
    neighbors and genre-similar artists' tracks, rather than brute-forcing
    the whole catalog against every seed."""
    candidates: set = set()

    in_vocab_seeds = [
        t for t in seed_track_ids if train_embeddings.is_in_vocabulary(kv, t)
    ]
    if in_vocab_seeds:
        neighbors = kv.most_similar(
            positive=in_vocab_seeds, topn=CANDIDATE_TOPN_EMBEDDING
        )
        candidates.update(track_id for track_id, _ in neighbors)

    for seed_artist in seed_artist_ids:
        similar_artists = genre_similarity.most_similar_artists(
            seed_artist, artist_ids, sim_matrix, top_n=CANDIDATE_TOPN_GENRE_ARTISTS
        )
        for similar_artist, _ in similar_artists:
            candidates.update(artist_track_index.get(similar_artist, []))

    return candidates - seed_track_ids


# --- Scoring ---------------------------------------------------------------


def _min_max_normalize(values: list) -> list:
    if not values:
        return []
    lo, hi = min(values), max(values)
    if hi == lo:
        return [1.0 for _ in values]
    return [(v - lo) / (hi - lo) for v in values]


def explain(
    signal_embedding: float | None, signal_genre: float | None, weight_embedding: float
) -> str:
    if signal_embedding is not None and signal_genre is not None:
        return (
            f"Matched on listening co-occurrence (similarity={signal_embedding:.2f}) "
            f"and genre overlap (similarity={signal_genre:.2f}), blended "
            f"{weight_embedding:.0%}/{1 - weight_embedding:.0%}."
        )
    if signal_embedding is not None:
        return (
            f"Matched via listening co-occurrence only (similarity={signal_embedding:.2f}) "
            "-- no genre data available for this artist."
        )
    return (
        f"Matched via genre similarity only (similarity={signal_genre:.2f}) -- track "
        "appears too rarely in the training corpus for an embedding (cold-start fallback)."
    )


def score_candidates(
    seed_track_ids: set,
    seed_artist_ids: set,
    candidate_ids: set,
    kv: KeyedVectors,
    catalog: dict,
    artist_ids: list,
    sim_matrix: np.ndarray,
    weight_embedding: float = config.DEFAULT_WEIGHT_EMBEDDING,
) -> list:
    """Pure, deterministic, no I/O -- independently testable without the LLM."""
    in_vocab_seeds = [
        t for t in seed_track_ids if train_embeddings.is_in_vocabulary(kv, t)
    ]

    raw = []  # (candidate_id, signal1, signal2)
    for candidate_id in candidate_ids:
        info = catalog.get(candidate_id)
        if info is None:
            continue

        signal1 = None
        if in_vocab_seeds and train_embeddings.is_in_vocabulary(kv, candidate_id):
            signal1 = float(
                np.mean([kv.similarity(candidate_id, s) for s in in_vocab_seeds])
            )

        signal2 = None
        candidate_artist = info["artist_id"]
        artist_scores = [
            score
            for seed_artist in seed_artist_ids
            if (
                score := genre_similarity.artist_similarity(
                    candidate_artist, seed_artist, artist_ids, sim_matrix
                )
            )
            is not None
        ]
        if artist_scores:
            signal2 = float(np.mean(artist_scores))

        if signal1 is None and signal2 is None:
            continue
        raw.append((candidate_id, signal1, signal2))

    idx1 = [i for i, (_, s1, _) in enumerate(raw) if s1 is not None]
    idx2 = [i for i, (_, _, s2) in enumerate(raw) if s2 is not None]
    norm1_map = dict(zip(idx1, _min_max_normalize([raw[i][1] for i in idx1])))
    norm2_map = dict(zip(idx2, _min_max_normalize([raw[i][2] for i in idx2])))

    scored = []
    for i, (candidate_id, signal1, signal2) in enumerate(raw):
        n1, n2 = norm1_map.get(i), norm2_map.get(i)
        if n1 is not None and n2 is not None:
            final = weight_embedding * n1 + (1 - weight_embedding) * n2
        elif n1 is not None:
            final = n1
        else:
            final = n2

        info = catalog[candidate_id]
        scored.append(
            ScoredTrack(
                track_id=candidate_id,
                name=info["name"],
                artist_id=info["artist_id"],
                artist_name=info["artist_name"],
                signal_embedding=signal1,
                signal_genre=signal2,
                final_score=final,
                why=explain(signal1, signal2, weight_embedding),
            )
        )

    scored.sort(key=lambda t: t.final_score, reverse=True)
    return scored


# --- Pipeline entry point ---------------------------------------------------


def _load_pipeline_artifacts():
    kv = train_embeddings.load_track_vectors()
    catalog = train_embeddings.load_track_catalog()
    artist_ids, sim_matrix = genre_similarity.load_genre_model()
    artist_track_index = build_artist_track_index(catalog)
    return kv, catalog, artist_ids, sim_matrix, artist_track_index


def recommend(
    time_window: str = "all_time",
    genre_filter: str | None = None,
    weight_embedding: float = config.DEFAULT_WEIGHT_EMBEDDING,
    top_n: int = 10,
) -> list:
    kv, catalog, artist_ids, sim_matrix, artist_track_index = _load_pipeline_artifacts()
    seed = build_seed_profile(time_window)

    candidates = generate_candidates(
        seed.track_ids, seed.artist_ids, kv, artist_ids, sim_matrix, artist_track_index
    )

    if genre_filter:
        artist_genres = fetch.load_artist_genres()
        genre_filter_lower = genre_filter.lower()
        candidates = {
            c
            for c in candidates
            if genre_filter_lower
            in [
                g.lower()
                for g in artist_genres.get(catalog[c]["artist_id"], {}).get(
                    "genres", []
                )
            ]
        }

    scored = score_candidates(
        seed.track_ids,
        seed.artist_ids,
        candidates,
        kv,
        catalog,
        artist_ids,
        sim_matrix,
        weight_embedding,
    )
    return scored[:top_n]


# --- NL query layer ---------------------------------------------------------


def extract_intent(query: str, known_genres: list) -> IntentResult:
    try:
        client = genai.Client(api_key=config.require_env("GEMINI_API_KEY"))
        response = client.models.generate_content(
            model=INTENT_MODEL,
            contents=query,
            config=genai_types.GenerateContentConfig(
                system_instruction=INTENT_SYSTEM_PROMPT,
                tools=[genai_types.Tool(function_declarations=[INTENT_TOOL])],
                tool_config=genai_types.ToolConfig(
                    function_calling_config=genai_types.FunctionCallingConfig(
                        mode=genai_types.FunctionCallingConfigMode.ANY,
                        allowed_function_names=["record_intent"],
                    )
                ),
            ),
        )
        function_call = response.candidates[0].content.parts[0].function_call
        data = function_call.args
        return IntentResult(
            genre=data.get("genre"),
            time_window=data.get("time_window"),
            needs_clarification=data.get("needs_clarification", False),
            clarifying_question=data.get("clarifying_question"),
        )
    except Exception:
        logger.warning("extract_intent failed, asking user to rephrase", exc_info=True)
        return IntentResult(
            genre=None,
            time_window=None,
            needs_clarification=True,
            clarifying_question=(
                "Sorry, I couldn't process that -- could you rephrase, mentioning "
                "a genre and/or a time period?"
            ),
        )


def validate_genre(genre: str | None, known_genres: list) -> tuple[str | None, list]:
    """Fuzzy-matches the extracted genre against genres actually seen in the
    user's data -- the LLM can't see the user's real genre vocabulary, so
    this check happens in code, not the prompt."""
    if genre is None:
        return None, []
    if genre in known_genres:
        return genre, []
    alternatives = difflib.get_close_matches(genre, known_genres, n=3, cutoff=0.6)
    return None, alternatives


def _print_results(results: list) -> None:
    if not results:
        print("No recommendations found -- try a broader time window or no genre filter.")
        return
    for i, track in enumerate(results, start=1):
        print(f"{i}. {track.name} -- {track.artist_name}  (score={track.final_score:.3f})")
        print(f"   {track.why}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", nargs="?", default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument(
        "--weight-embedding", type=float, default=config.DEFAULT_WEIGHT_EMBEDDING
    )
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    if args.no_llm:
        results = recommend(weight_embedding=args.weight_embedding, top_n=args.top_n)
        _print_results(results)
        return

    query = args.query or input("What would you like recommendations for? ")
    known_genres = sorted(
        {g for info in fetch.load_artist_genres().values() for g in info.get("genres", [])}
    )

    time_window, genre = "all_time", None
    for _ in range(3):
        intent = extract_intent(query, known_genres)
        if not intent.needs_clarification:
            time_window = intent.time_window or "all_time"
            genre = intent.genre
            break
        print(intent.clarifying_question or "Could you clarify your request?")
        query = input("> ")
    else:
        print("Couldn't pin down a clear genre/time window -- using all-time, no genre filter.")

    if genre:
        matched, alternatives = validate_genre(genre, known_genres)
        if matched is None:
            if alternatives:
                print(
                    f"'{genre}' isn't a genre I've seen in your history. "
                    f"Did you mean: {', '.join(alternatives)}? Proceeding without a genre filter."
                )
            else:
                print(
                    f"'{genre}' isn't a genre I've seen in your history -- "
                    "proceeding without a genre filter."
                )
            genre = None
        else:
            genre = matched

    print(f"Parsed intent -- genre: {genre or 'not detected'}, time window: {time_window}")
    if not args.yes:
        confirm = input("Proceed? [Y/n] ").strip().lower()
        if confirm not in ("", "y", "yes"):
            print("Cancelled.")
            return

    results = recommend(
        time_window=time_window,
        genre_filter=genre,
        weight_embedding=args.weight_embedding,
        top_n=args.top_n,
    )
    _print_results(results)


if __name__ == "__main__":
    main()
