"""Fetches Spotify listening data and locates the local MPD corpus.

Raw Spotify API responses are cached as JSON under data/raw/spotify/ so
reruns read the cache instead of re-hitting rate limits (CLAUDE.md's
caching rule). The Million Playlist Dataset is NOT downloaded by this
module -- download it manually from
https://www.kaggle.com/datasets/himanshuwagh/spotify-million and extract
its mpd.slice.*.json files into data/raw/mpd/. This module only locates
and parses whatever is already there.

Note on scope: as of Spotify's Nov 2024 + Feb 2026 Web API changes, apps
created after Nov 27 2024 (this one) have no live path to a public
playlist corpus (Featured/Category Playlists and Related Artists are
removed, and other users' playlist contents are restricted to metadata
only) -- hence the offline MPD dependency instead of a live crawl.
"""

import argparse
import json
import time
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path

import config

logger = config.get_logger(__name__)

TIME_RANGES = ("short_term", "medium_term", "long_term")
SAVED_TRACKS_PAGE_SIZE = 50


# --- MPD locating (not downloading) -------------------------------------


def load_mpd_slice_paths(mpd_dir: Path = config.MPD_DIR) -> list[Path]:
    """Finds MPD slice files under `mpd_dir`, sorted by slice start index."""
    paths = sorted(
        mpd_dir.rglob("mpd.slice.*.json"),
        key=lambda p: int(p.name.split(".")[2].split("-")[0]),
    )
    if not paths:
        raise FileNotFoundError(
            f"No MPD slice files found under {mpd_dir}. Download the Million "
            "Playlist Dataset (https://www.kaggle.com/datasets/himanshuwagh/"
            f"spotify-million) and extract its mpd.slice.*.json files into {mpd_dir}."
        )
    return paths


def iter_mpd_slices(paths: list[Path]) -> Iterator[dict]:
    for path in paths:
        with path.open("r", encoding="utf-8") as f:
            yield json.load(f)


# --- Generic JSON cache helpers ------------------------------------------


def _cache_path(name: str) -> Path:
    return config.SPOTIFY_RAW_DIR / f"{name}.json"


def _read_cache(name: str) -> dict | None:
    path = _cache_path(name)
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _write_cache(name: str, data: dict) -> None:
    config.SPOTIFY_RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = _cache_path(name)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    logger.info("Wrote %s", path)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_tracks(name: str) -> list[dict]:
    """Loads a cached Spotify JSON file and returns a flat list of track
    objects. Handles both directly-listed items (top_tracks_*) and the
    {"track": {...}} wrapper used by saved_tracks/recently_played."""
    data = _read_cache(name) or {}
    items = data.get("items", [])
    return [entry.get("track", entry) for entry in items if entry]


def load_artist_genres() -> dict:
    return _read_cache("artist_genres") or {}


# --- Spotify pulls --------------------------------------------------------


def fetch_top_tracks(sp, force_refresh: bool = False) -> dict:
    """Returns {time_range: [track, ...]}."""
    result = {}
    for time_range in TIME_RANGES:
        name = f"top_tracks_{time_range}"
        cached = None if force_refresh else _read_cache(name)
        if cached:
            logger.info("Using cached %s", name)
            result[time_range] = cached["items"]
            continue
        response = sp.current_user_top_tracks(limit=50, time_range=time_range)
        items = response.get("items", [])
        _write_cache(
            name, {"fetched_at": _now_iso(), "time_range": time_range, "items": items}
        )
        result[time_range] = items
    return result


def fetch_top_artists(sp, force_refresh: bool = False) -> dict:
    """Returns {time_range: [artist, ...]}. Artist objects already embed genres."""
    result = {}
    for time_range in TIME_RANGES:
        name = f"top_artists_{time_range}"
        cached = None if force_refresh else _read_cache(name)
        if cached:
            logger.info("Using cached %s", name)
            result[time_range] = cached["items"]
            continue
        response = sp.current_user_top_artists(limit=50, time_range=time_range)
        items = response.get("items", [])
        _write_cache(
            name, {"fetched_at": _now_iso(), "time_range": time_range, "items": items}
        )
        result[time_range] = items
    return result


def fetch_saved_tracks(sp, max_pages: int = 10, force_refresh: bool = False) -> dict:
    cached = None if force_refresh else _read_cache("saved_tracks")
    if cached:
        logger.info("Using cached saved_tracks")
        return cached

    items = []
    offset = 0
    for _ in range(max_pages):
        response = sp.current_user_saved_tracks(
            limit=SAVED_TRACKS_PAGE_SIZE, offset=offset
        )
        page_items = response.get("items", [])
        items.extend(page_items)
        if len(page_items) < SAVED_TRACKS_PAGE_SIZE:
            break
        offset += SAVED_TRACKS_PAGE_SIZE
    data = {"fetched_at": _now_iso(), "items": items}
    _write_cache("saved_tracks", data)
    return data


def fetch_recently_played(sp, force_refresh: bool = False) -> dict:
    # Spotify hard-caps this at 50 with no pagination (CLAUDE.md constraint).
    cached = None if force_refresh else _read_cache("recently_played")
    if cached:
        logger.info("Using cached recently_played")
        return cached

    response = sp.current_user_recently_played(limit=50)
    data = {"fetched_at": _now_iso(), "items": response.get("items", [])}
    _write_cache("recently_played", data)
    return data


def fetch_artist_genres(
    sp,
    top_artists: dict,
    track_artist_ids: set,
    force_refresh: bool = False,
) -> dict:
    """Builds artist_id -> {name, genres, fetched_at}.

    Seeds from top-artists responses (which already embed genres), then
    fetches any remaining artist IDs one at a time via GET /artists/{id} --
    batch lookups (/v1/artists?ids=) are treated as unreliable under
    current Spotify Development Mode restrictions. Reruns only fetch IDs
    not already cached (genres rarely change; delete artist_genres.json
    to force a full refresh).
    """
    existing = {} if force_refresh else _read_cache("artist_genres") or {}
    genre_map = dict(existing)

    for artists in top_artists.values():
        for artist in artists:
            genre_map[artist["id"]] = {
                "name": artist.get("name"),
                "genres": artist.get("genres", []),
                "fetched_at": _now_iso(),
            }

    missing = [aid for aid in track_artist_ids if aid not in genre_map]
    logger.info("Fetching genres for %d artists not already cached", len(missing))
    for artist_id in missing:
        try:
            artist = sp.artist(artist_id)
        except Exception:
            logger.warning(
                "Failed to fetch artist %s, skipping", artist_id, exc_info=True
            )
            continue
        genre_map[artist_id] = {
            "name": artist.get("name"),
            "genres": artist.get("genres", []),
            "fetched_at": _now_iso(),
        }
        time.sleep(0.05)

    _write_cache("artist_genres", genre_map)
    return genre_map


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--skip-genres", action="store_true")
    parser.add_argument("--max-saved-pages", type=int, default=10)
    args = parser.parse_args()

    import auth  # local import: only needed when actually hitting the API

    sp = auth.get_spotify_client()

    top_tracks = fetch_top_tracks(sp, force_refresh=args.force_refresh)
    top_artists = fetch_top_artists(sp, force_refresh=args.force_refresh)
    saved_tracks = fetch_saved_tracks(
        sp, max_pages=args.max_saved_pages, force_refresh=args.force_refresh
    )
    recently_played = fetch_recently_played(sp, force_refresh=args.force_refresh)

    if not args.skip_genres:
        track_artist_ids = set()
        for name in (
            "top_tracks_short_term",
            "top_tracks_medium_term",
            "top_tracks_long_term",
            "saved_tracks",
            "recently_played",
        ):
            for track in load_tracks(name):
                for artist in track.get("artists", []):
                    if artist.get("id"):
                        track_artist_ids.add(artist["id"])
        genre_map = fetch_artist_genres(
            sp, top_artists, track_artist_ids, force_refresh=args.force_refresh
        )
        logger.info("Genre map covers %d artists", len(genre_map))

    logger.info(
        "Done. top_tracks ranges=%d, top_artists ranges=%d, saved_tracks=%d, "
        "recently_played=%d",
        len(top_tracks),
        len(top_artists),
        len(saved_tracks.get("items", [])),
        len(recently_played.get("items", [])),
    )


if __name__ == "__main__":
    main()
