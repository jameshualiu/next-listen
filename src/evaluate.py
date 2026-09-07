"""Holdout evaluation: precision@10 / recall@10 vs. a random baseline,
swept across a range of Signal-1/Signal-2 blend weights.

Calls recommend.py's scoring core (generate_candidates/score_candidates)
directly -- never extract_intent() or the CLI confirmation flow -- so
evaluation is deterministic, free of LLM calls, and safe to run in CI.

If no real data/processed artifacts exist yet (e.g. a fresh checkout in
CI, with no Spotify credentials and data/ gitignored), this runs a smoke
test against the small committed fixtures in tests/fixtures/ instead,
labeled "mode": "fixture" in the report -- see .github/workflows/evaluate.yml.

Run directly:

    python src/evaluate.py
"""

import json
import random
from datetime import datetime, timezone

import config
import fetch
import genre_similarity
import recommend
import train_embeddings

logger = config.get_logger(__name__)

LIKED_SOURCES = [
    "saved_tracks",
    "top_tracks_short_term",
    "top_tracks_medium_term",
    "top_tracks_long_term",
]
WEIGHT_SWEEP = [0.0, 0.25, 0.5, 0.6, 0.75, 1.0]
RANDOM_TRIALS = 100
TOP_N = 10
FIXTURES_DIR = config.PROJECT_ROOT / "tests" / "fixtures"


def _train_test_split(liked_ids: set, seed: int = config.RANDOM_SEED) -> tuple:
    ids = sorted(liked_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    split_point = int(len(ids) * 0.8)
    return set(ids[:split_point]), set(ids[split_point:])


def _precision_recall(recommended_ids: list, holdout: set) -> tuple:
    hit_count = len(set(recommended_ids) & holdout)
    precision = hit_count / len(recommended_ids) if recommended_ids else 0.0
    recall = hit_count / len(holdout) if holdout else 0.0
    return precision, recall


def _random_baseline(
    catalog: dict, exclude: set, holdout: set, seed: int = config.RANDOM_SEED
) -> tuple:
    rng = random.Random(seed)
    pool = [t for t in catalog if t not in exclude]
    precisions, recalls = [], []
    for _ in range(RANDOM_TRIALS):
        sample = rng.sample(pool, min(TOP_N, len(pool))) if pool else []
        p, r = _precision_recall(sample, holdout)
        precisions.append(p)
        recalls.append(r)
    return sum(precisions) / len(precisions), sum(recalls) / len(recalls)


def _weight_sweep(
    train_ids: set,
    train_artist_ids: set,
    holdout_ids: set,
    kv,
    catalog: dict,
    artist_ids: list,
    sim_matrix,
    artist_track_index: dict,
) -> list:
    candidates = recommend.generate_candidates(
        train_ids, train_artist_ids, kv, artist_ids, sim_matrix, artist_track_index
    )
    sweep_results = []
    for weight in WEIGHT_SWEEP:
        scored = recommend.score_candidates(
            train_ids, train_artist_ids, candidates, kv, catalog, artist_ids, sim_matrix, weight
        )
        top_ids = [s.track_id for s in scored[:TOP_N]]
        precision, recall = _precision_recall(top_ids, holdout_ids)
        sweep_results.append(
            {"weight_embedding": weight, "precision_at_10": precision, "recall_at_10": recall}
        )
    return sweep_results


# --- Real mode --------------------------------------------------------------


def _has_real_artifacts() -> bool:
    return (
        (config.DATA_PROCESSED / "track_vectors.kv").exists()
        and (config.DATA_PROCESSED / "track_catalog.json").exists()
        and (config.DATA_PROCESSED / "genre_similarity_matrix.npz").exists()
        and (config.SPOTIFY_RAW_DIR / "saved_tracks.json").exists()
    )


def _load_liked_track_ids() -> set:
    liked = set()
    for source in LIKED_SOURCES:
        for track in fetch.load_tracks(source):
            if track.get("id"):
                liked.add(track["id"])
    return liked


def run_real() -> dict:
    kv = train_embeddings.load_track_vectors()
    catalog = train_embeddings.load_track_catalog()
    artist_ids, sim_matrix = genre_similarity.load_genre_model()
    artist_track_index = recommend.build_artist_track_index(catalog)

    liked_ids = _load_liked_track_ids()
    train_ids, holdout_ids = _train_test_split(liked_ids)
    train_artist_ids = {catalog[t]["artist_id"] for t in train_ids if t in catalog}

    sweep_results = _weight_sweep(
        train_ids, train_artist_ids, holdout_ids, kv, catalog, artist_ids, sim_matrix, artist_track_index
    )
    baseline_precision, baseline_recall = _random_baseline(catalog, train_ids, holdout_ids)

    return {
        "mode": "real",
        "liked_count": len(liked_ids),
        "train_count": len(train_ids),
        "holdout_count": len(holdout_ids),
        "weight_sweep": sweep_results,
        "random_baseline": {
            "precision_at_10": baseline_precision,
            "recall_at_10": baseline_recall,
        },
    }


# --- Fixture mode (keeps CI green without credentials or real data) --------


def run_fixture() -> dict:
    with (FIXTURES_DIR / "mpd_playlists.json").open("r", encoding="utf-8") as f:
        slice_data = json.load(f)
    sentences, catalog = train_embeddings.build_corpus_from_slices([slice_data])
    model = train_embeddings.train_word2vec(
        sentences, vector_size=16, window=5, min_count=1, epochs=5
    )
    kv = model.wv

    with (FIXTURES_DIR / "artist_genres.json").open("r", encoding="utf-8") as f:
        artist_genres = json.load(f)
    artist_ids, sim_matrix, _ = genre_similarity.compute_similarity(artist_genres)

    with (FIXTURES_DIR / "liked_tracks.json").open("r", encoding="utf-8") as f:
        liked_tracks = json.load(f)
    liked_ids = {t["id"] for t in liked_tracks if t.get("id")}

    train_ids, holdout_ids = _train_test_split(liked_ids)
    train_artist_ids = {catalog[t]["artist_id"] for t in train_ids if t in catalog}
    artist_track_index = recommend.build_artist_track_index(catalog)

    candidates = recommend.generate_candidates(
        train_ids, train_artist_ids, kv, artist_ids, sim_matrix, artist_track_index
    )
    scored = recommend.score_candidates(
        train_ids,
        train_artist_ids,
        candidates,
        kv,
        catalog,
        artist_ids,
        sim_matrix,
        config.DEFAULT_WEIGHT_EMBEDDING,
    )
    top_ids = [s.track_id for s in scored[:TOP_N]]
    precision, recall = _precision_recall(top_ids, holdout_ids)

    return {
        "mode": "fixture",
        "note": (
            "No real data/processed artifacts found -- ran a smoke test against "
            "committed fixtures in tests/fixtures/ instead of real Spotify data."
        ),
        "liked_count": len(liked_ids),
        "train_count": len(train_ids),
        "holdout_count": len(holdout_ids),
        "precision_at_10": precision,
        "recall_at_10": recall,
    }


# --- Reporting ---------------------------------------------------------------


def _render_markdown(report: dict) -> str:
    lines = [f"# Evaluation report ({report['mode']} mode)", "", f"Generated: {report['generated_at']}", ""]
    if report["mode"] == "real":
        lines += [
            f"- Liked tracks: {report['liked_count']}",
            f"- Train: {report['train_count']}, Holdout: {report['holdout_count']}",
            "",
            "| weight_embedding | precision@10 | recall@10 |",
            "|---|---|---|",
        ]
        for row in report["weight_sweep"]:
            lines.append(
                f"| {row['weight_embedding']} | {row['precision_at_10']:.3f} | {row['recall_at_10']:.3f} |"
            )
        baseline = report["random_baseline"]
        lines += [
            "",
            f"Random baseline (avg of {RANDOM_TRIALS} trials): "
            f"precision@10={baseline['precision_at_10']:.3f}, recall@10={baseline['recall_at_10']:.3f}",
        ]
    else:
        lines += [
            report["note"],
            "",
            f"- Liked tracks: {report['liked_count']}",
            f"- Train: {report['train_count']}, Holdout: {report['holdout_count']}",
            f"- precision@10: {report['precision_at_10']:.3f}",
            f"- recall@10: {report['recall_at_10']:.3f}",
        ]
    return "\n".join(lines) + "\n"


def _write_report(report: dict) -> None:
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_with_meta = {"generated_at": datetime.now(timezone.utc).isoformat(), **report}
    md_content = _render_markdown(report_with_meta)

    for name, is_json in (
        (f"eval_report_{timestamp}.json", True),
        (f"eval_report_{timestamp}.md", False),
        ("latest.json", True),
        ("latest.md", False),
    ):
        path = config.RESULTS_DIR / name
        if is_json:
            with path.open("w", encoding="utf-8") as f:
                json.dump(report_with_meta, f, indent=2)
        else:
            path.write_text(md_content, encoding="utf-8")

    logger.info("Wrote results/eval_report_%s.{json,md} and results/latest.{json,md}", timestamp)


def main() -> None:
    if _has_real_artifacts():
        logger.info("Real data/processed artifacts found -- running full evaluation.")
        report = run_real()
    else:
        logger.info("No real artifacts found -- running fixture-mode smoke test.")
        report = run_fixture()
    _write_report(report)


if __name__ == "__main__":
    main()
